// ======================================================
// XIAO ML KIT (OR XIAO ESP32S3 SENSE)
// VISION ML - ANN + SNN INFERENCE ONLY - v046
//
// Companion firmware to the browser trainer (index-vision-v001.html).
// All training now happens in the browser; this firmware COLLECTS images
// and RUNS INFERENCE with two classifiers trained on the same architecture:
//   ANN  Conv1(3x3)->Pool->Conv2(3x3)->Dense->softmax, ordinary neurons.
//   SNN  Same conv/pool/dense SHAPE, but every neuron is a spiking LIF
//        neuron. A camera frame has no real time axis, so SNN_TIMESTEPS
//        synthetic steps are used: each step independently rate-codes the
//        image into a fresh Bernoulli spike frame (pixel value = spike
//        probability), and the two conv layers integrate spikes over those
//        steps via leaky membrane potentials. A per-class leaky trace,
//        read off the flattened Conv2 spikes, accumulates over all steps;
//        its value after the last step is softmaxed for the prediction.
//        (Trained in the browser via surrogate-gradient BPTT through the
//        synthetic timesteps - see index-vision-v001.html.)
//
// WHAT CHANGED FROM v44 (on-device-vision-ai)
//   - On-device TRAINING REMOVED (myBackward*, myAdamUpdate, myUpdateWeights,
//     myActionTrain, myLoadImageFromFile all deleted - they were training-only).
//     Train in the browser instead; drop the two SD card headers here.
//   - SNN inference ADDED. Same architecture shapes as the ANN (CONV1_FILTERS,
//     CONV2_FILTERS, FLATTENED_SIZE, OUTPUT_WEIGHTS are shared #defines), so
//     the ANN and SNN weight buffers are the same sizes - only the neuron
//     model and the SNN's extra time loop differ.
//   - INPUT_SIZE and NUM_CHANNELS are both now real settings, not just
//     INPUT_SIZE. NUM_CHANNELS=3 is RGB (original behaviour), NUM_CHANNELS=1
//     is grayscale (luminance-converted from the color sensor at preprocess
//     time - the camera and JPEG files on SD are unaffected either way).
//     Whichever combination you train in the browser, set the SAME two
//     #defines here before flashing.
//   - LIF_LEAK / LIF_THRESHOLD / SNN_TRACE_LEAK / SNN_TIMESTEPS are NOT
//     stored in the weight file (same convention as the motion firmware) -
//     they must match whatever the browser trainer used. Check the
//     browser's Status panel after training and copy its values here.
//   - Menu's last two items are now "Infer ANN" and "Infer SNN" (was
//     "Train" and "Infer").
//
// SD card stores: images in class folders (unchanged)
// SD card reads:  /header/myWeights.bin (ANN), /header/mySnnWeights.bin (SNN)
// Serial monitor and OLED output
// By Jeremy Ellis
// With free tier assistance from: Claude (code overview), ChatGPT (Critique), Gemini (Research) and Copilot (Alternate)
// Use at your own risk!
// MIT license
//
// Github Profile https://github.com/hpssjellis
// LinkedIn https://www.linkedin.com/in/jeremy-ellis-4237a9bb/
//
// For platformio you need the U8g2 library declared in the platformio.ini file and OPI PSRAM set
//   lib_deps = olikraus/U8g2 @ ^2.35.30
//   build_flags = -DBOARD_HAS_PSRAM -DARDUINO_USB_CDC_ON_BOOT=1
//   board_build.arduino.memory_type = qio_opi
//   board_build.flash_mode = qio
//   board_upload.flash_size = 8MB
//
// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 0: CORE SYSTEM (ALWAYS INCLUDED)                                   ██
// ██  Headers, Defines, Pins, Globals, Memory, Weights, Setup, Loop           ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████

// optional Uncomment AFTER copying myWeights.h / mySnnWeights.h from SD to your sketch folder:
// Priority order: SD weights > baked-in weights > random He-init
//////////////////////////////////////IMPORTANT/////////////////////////////////////////////////
//#define USE_BAKED_WEIGHTS
#ifdef USE_BAKED_WEIGHTS
  #include "myWeights.h"
  #include "mySnnWeights.h"
#endif

#include "esp_camera.h"
#include "img_converters.h"
#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include <vector>
#include <algorithm>
#include <U8g2lib.h>
#include <Wire.h>

U8G2_SSD1306_72X40_ER_1_HW_I2C u8g2(U8G2_R2, U8X8_PIN_NONE);

// ======================================================
// CONFIGURATION - classes must match whatever you trained in the browser
// ======================================================
#define NUM_CLASSES 3
String myClassLabels[NUM_CLASSES] = {"0Blank", "1Cup", "2Pen"};
const int myTotalItems = NUM_CLASSES + 2;   // NUM_CLASSES + Infer ANN + Infer SNN

const int myThresholdPress = 1100;
const int myThresholdRelease = 900;

// ======================================================
// UNIFIED TOUCH INPUT SYSTEM (unchanged from v44)
// ======================================================
struct TouchState {
  bool isTouching = false;
  int tapCount = 0;
  unsigned long firstTapTime = 0;
  unsigned long lastReleaseTime = 0;
  unsigned long lastCheckTime = 0;
  const unsigned long tapWindow = 800;
  const int longPressTaps = 3;
  const unsigned long debounceDelay = 50;
};
TouchState myTouch;

unsigned long myLastActivityTime = 0;
unsigned long myLastTapTime = 0;
const int myTapCooldown = 250;
int myMenuIndex = 1;
bool myIsSelected = false;
bool myAnnTrained = false;   // true once myWeights.bin loaded successfully
bool mySnnTrained = false;   // true once mySnnWeights.bin loaded successfully

// XIAO ESP32-S3 Camera Pins
#define PWDN_GPIO_NUM  -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  10
#define SIOD_GPIO_NUM  40
#define SIOC_GPIO_NUM  39
#define Y9_GPIO_NUM    48
#define Y8_GPIO_NUM    11
#define Y7_GPIO_NUM    12
#define Y6_GPIO_NUM    14
#define Y5_GPIO_NUM    16
#define Y4_GPIO_NUM    18
#define Y3_GPIO_NUM    17
#define Y2_GPIO_NUM    15
#define VSYNC_GPIO_NUM 38
#define HREF_GPIO_NUM  47
#define PCLK_GPIO_NUM  13

// ======================================================
// CONFIGURABLE INPUT RESOLUTION AND COLOR MODE
// Must match whatever the browser trainer used. NUM_CHANNELS=3 -> RGB,
// NUM_CHANNELS=1 -> grayscale (luminance-converted after camera capture -
// the camera sensor and the JPEGs saved to SD are unaffected either way).
// ======================================================
#define INPUT_SIZE 64
#define NUM_CHANNELS 3

// ======================================================
// CNN ARCHITECTURE CONSTANTS - shared by BOTH the ANN and the SNN.
// Only the neuron model (leaky-ReLU vs. spiking LIF) and the SNN's extra
// time loop differ; the weight buffer SHAPES are identical, which is what
// keeps the two models directly comparable.
// ======================================================
#define CONV1_KERNEL_SIZE 3
#define CONV1_FILTERS 4
#define CONV1_W_PER_FILTER (CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE * NUM_CHANNELS)
#define CONV1_WEIGHTS (CONV1_W_PER_FILTER * CONV1_FILTERS)

#define CONV2_KERNEL_SIZE 3
#define CONV2_FILTERS 8
#define CONV2_W_PER_FILTER (CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE * CONV1_FILTERS)
#define CONV2_WEIGHTS (CONV2_W_PER_FILTER * CONV2_FILTERS)

#define CONV1_OUTPUT_SIZE (INPUT_SIZE - (CONV1_KERNEL_SIZE - 1))
#define POOL1_OUTPUT_SIZE (CONV1_OUTPUT_SIZE / 2)
#define CONV2_OUTPUT_SIZE (POOL1_OUTPUT_SIZE - (CONV2_KERNEL_SIZE - 1))
#define FLATTENED_SIZE (CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE * CONV2_FILTERS)
#define OUTPUT_WEIGHTS (FLATTENED_SIZE * NUM_CLASSES)

// ======================================================
// SNN-ONLY SETTINGS
// Synthetic timesteps used to rate-code one still frame, and the spiking
// neuron dynamics. NONE of these are stored in mySnnWeights.bin (same
// convention as the motion firmware) - they must match the browser's
// training settings exactly, or the loaded weights will behave differently
// than they did when trained. Check the browser's Status panel.
// ======================================================
#define SNN_TIMESTEPS 16
float LIF_LEAK = 0.90f;
float LIF_THRESHOLD = 0.50f;
float SNN_TRACE_LEAK = 0.95f;

// ======================================================
// GLOBAL VARIABLE DEFINITIONS
// ======================================================
uint8_t* myRgbBuffer = nullptr;     // reusable 240x240x3 RGB buffer
bool mySDavailable = false;

// ---- ANN weights (PSRAM) ----
float* myInputBuffer = nullptr;     // normalized 0-1 pixels, NUM_CHANNELS per pixel
float* myConv1_w = nullptr;  float* myConv1_b = nullptr;
float* myConv2_w = nullptr;  float* myConv2_b = nullptr;
float* myOutput_w = nullptr; float* myOutput_b = nullptr;
float* myConv1_output = nullptr;
float* myPool1_output = nullptr;
float* myConv2_output = nullptr;
float* myDense_output = nullptr;    // ANN softmax probabilities [NUM_CLASSES]

// ---- SNN weights (PSRAM) - same shapes as the ANN's, different math ----
float* mySnnConv1_w = nullptr;  float* mySnnConv1_b = nullptr;
float* mySnnConv2_w = nullptr;  float* mySnnConv2_b = nullptr;
float* mySnnOutput_w = nullptr; float* mySnnOutput_b = nullptr;
float* mySnnDense_output = nullptr; // SNN softmax probabilities [NUM_CLASSES], after SNN_TIMESTEPS

// ---- SNN streaming/scratch buffers (persist membrane potential across timesteps) ----
float*   mySnnC1Mem = nullptr;      // [CONV1_FILTERS*CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE]
float*   mySnnC2Mem = nullptr;      // [CONV2_FILTERS*CONV2_OUTPUT_SIZE*CONV2_OUTPUT_SIZE]
float*   mySnnTrace = nullptr;      // [NUM_CLASSES]
uint8_t* mySnnInputSpike = nullptr; // [INPUT_SIZE*INPUT_SIZE*NUM_CHANNELS] - this timestep's rate-coded frame
uint8_t* mySnnC1Spike = nullptr;    // [CONV1_FILTERS*CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE]
uint8_t* mySnnPooled = nullptr;     // [CONV1_FILTERS*POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE]
uint8_t* mySnnC2Spike = nullptr;    // [FLATTENED_SIZE] (CONV2_FILTERS*CONV2_OUTPUT_SIZE*CONV2_OUTPUT_SIZE)

// ======================================================
// UTILITY FUNCTIONS
// ======================================================
inline float clip_value(float v, float mn=-100, float mx=100) {
  if (isnan(v) || isinf(v)) return 0;
  return constrain(v, mn, mx);
}
inline float leaky_relu(float x) { return x > 0 ? x : 0.1f * x; }

// ======================================================
// UNIFIED TOUCH INPUT FUNCTIONS (unchanged from v44)
// ======================================================
int myReadTouch() {
  int sum = 0;
  for (int i = 0; i < 3; i++) { sum += analogRead(A0); delayMicroseconds(100); }
  return sum / 3;
}
void myResetTouchState() {
  myTouch.isTouching = false; myTouch.tapCount = 0;
  myTouch.firstTapTime = 0; myTouch.lastReleaseTime = 0; myTouch.lastCheckTime = 0;
}
void myUpdateTouchState() {
  unsigned long now = millis();
  if (now - myTouch.lastCheckTime < 20) return;
  myTouch.lastCheckTime = now;
  int val = myReadTouch();
  bool touchActive = myTouch.isTouching ? (val > myThresholdRelease) : (val > myThresholdPress);
  if (touchActive && !myTouch.isTouching) {
    if (now - myTouch.lastReleaseTime < myTouch.debounceDelay) return;
    myTouch.isTouching = true;
    if (myTouch.tapCount == 0 || (now - myTouch.firstTapTime < myTouch.tapWindow)) {
      if (myTouch.tapCount == 0) myTouch.firstTapTime = now;
      myTouch.tapCount++;
      Serial.printf("Tap #%d\n", myTouch.tapCount);
    } else {
      myTouch.tapCount = 1; myTouch.firstTapTime = now;
      Serial.println("Tap #1 (new window)");
    }
  }
  if (!touchActive && myTouch.isTouching) { myTouch.isTouching = false; myTouch.lastReleaseTime = now; }
}
int myCheckTouchInput() {
  myUpdateTouchState();
  unsigned long now = millis();
  if (myTouch.tapCount > 0 && !myTouch.isTouching) {
    if (now - myTouch.firstTapTime > myTouch.tapWindow) {
      int result = (myTouch.tapCount >= myTouch.longPressTaps) ? 2 : 1;
      int count = myTouch.tapCount;
      myResetTouchState();
      if (result == 2) Serial.printf("LONG PRESS detected (%d taps)\n", count);
      else Serial.printf("TAP detected (%d tap%s)\n", count, count > 1 ? "s" : "");
      return result;
    }
  }
  return 0;
}
void myCheckTouchBackground() { myUpdateTouchState(); }

// ======================================================
// MEMORY ALLOCATION
// ======================================================
void myAllocateMemory() {
  if (myInputBuffer != nullptr) return;
  Serial.println("\n=== Allocating Memory ===");

  myInputBuffer = (float*)ps_malloc(INPUT_SIZE * INPUT_SIZE * NUM_CHANNELS * sizeof(float));

  myConv1_w  = (float*)ps_malloc(CONV1_WEIGHTS * sizeof(float));
  myConv1_b  = (float*)ps_malloc(CONV1_FILTERS * sizeof(float));
  myConv2_w  = (float*)ps_malloc(CONV2_WEIGHTS * sizeof(float));
  myConv2_b  = (float*)ps_malloc(CONV2_FILTERS * sizeof(float));
  myOutput_w = (float*)ps_malloc(OUTPUT_WEIGHTS * sizeof(float));
  myOutput_b = (float*)ps_malloc(NUM_CLASSES * sizeof(float));

  myConv1_output = (float*)ps_malloc(CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE*CONV1_FILTERS*sizeof(float));
  myPool1_output = (float*)ps_malloc(POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE*CONV1_FILTERS*sizeof(float));
  myConv2_output = (float*)ps_malloc(CONV2_OUTPUT_SIZE*CONV2_OUTPUT_SIZE*CONV2_FILTERS*sizeof(float));
  myDense_output = (float*)ps_malloc(NUM_CLASSES*sizeof(float));

  mySnnConv1_w  = (float*)ps_malloc(CONV1_WEIGHTS * sizeof(float));
  mySnnConv1_b  = (float*)ps_malloc(CONV1_FILTERS * sizeof(float));
  mySnnConv2_w  = (float*)ps_malloc(CONV2_WEIGHTS * sizeof(float));
  mySnnConv2_b  = (float*)ps_malloc(CONV2_FILTERS * sizeof(float));
  mySnnOutput_w = (float*)ps_malloc(OUTPUT_WEIGHTS * sizeof(float));
  mySnnOutput_b = (float*)ps_malloc(NUM_CLASSES * sizeof(float));
  mySnnDense_output = (float*)ps_malloc(NUM_CLASSES*sizeof(float));

  mySnnC1Mem      = (float*)ps_calloc(CONV1_FILTERS*CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE, sizeof(float));
  mySnnC2Mem      = (float*)ps_calloc(CONV2_FILTERS*CONV2_OUTPUT_SIZE*CONV2_OUTPUT_SIZE, sizeof(float));
  mySnnTrace      = (float*)ps_calloc(NUM_CLASSES, sizeof(float));
  mySnnInputSpike = (uint8_t*)ps_malloc(INPUT_SIZE*INPUT_SIZE*NUM_CHANNELS*sizeof(uint8_t));
  mySnnC1Spike    = (uint8_t*)ps_malloc(CONV1_FILTERS*CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE*sizeof(uint8_t));
  mySnnPooled     = (uint8_t*)ps_malloc(CONV1_FILTERS*POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE*sizeof(uint8_t));
  mySnnC2Spike    = (uint8_t*)ps_malloc(FLATTENED_SIZE*sizeof(uint8_t));

  if (!myInputBuffer || !myConv1_w || !myConv2_w || !myOutput_w ||
      !myConv1_output || !myPool1_output || !myConv2_output ||
      !mySnnConv1_w || !mySnnConv2_w || !mySnnOutput_w || !mySnnC1Mem || !mySnnC2Mem) {
    Serial.println("FATAL: PSRAM allocation failed!");
    u8g2.firstPage(); do { u8g2.drawStr(0, 15, "PSRAM ERROR!"); } while (u8g2.nextPage());
    while (1) delay(1000);
  }
  Serial.printf("Free PSRAM after allocation: %d bytes\n", ESP.getFreePsram());

  // He-init both models with random weights (overwritten by SD/baked weights if present)
  float c1std = sqrt(2.0 / CONV1_W_PER_FILTER);
  for (int i=0;i<CONV1_WEIGHTS;i++){ myConv1_w[i]=((float)rand()/RAND_MAX-0.5f)*2*c1std; mySnnConv1_w[i]=((float)rand()/RAND_MAX-0.5f)*2*c1std; }
  for (int i=0;i<CONV1_FILTERS;i++){ myConv1_b[i]=0; mySnnConv1_b[i]=0; }
  float c2std = sqrt(2.0 / CONV2_W_PER_FILTER);
  for (int i=0;i<CONV2_WEIGHTS;i++){ myConv2_w[i]=((float)rand()/RAND_MAX-0.5f)*2*c2std; mySnnConv2_w[i]=((float)rand()/RAND_MAX-0.5f)*2*c2std; }
  for (int i=0;i<CONV2_FILTERS;i++){ myConv2_b[i]=0; mySnnConv2_b[i]=0; }
  float dstd = sqrt(2.0 / FLATTENED_SIZE);
  for (int i=0;i<OUTPUT_WEIGHTS;i++){ myOutput_w[i]=((float)rand()/RAND_MAX-0.5f)*2*dstd; mySnnOutput_w[i]=((float)rand()/RAND_MAX-0.5f)*2*dstd; }
  for (int i=0;i<NUM_CLASSES;i++){ myOutput_b[i]=0; mySnnOutput_b[i]=0; }
  Serial.println("He-init random weights set for ANN and SNN");
}

// ======================================================
// WEIGHT LOADING (read-only - training happens in the browser now)
// ======================================================
bool myLoadAnnWeights() {
  if (!mySDavailable) { Serial.println("No SD card - skipping ANN weight load"); return false; }
  if (!SD.exists("/header/myWeights.bin")) { Serial.println("No /header/myWeights.bin found"); return false; }
  File f = SD.open("/header/myWeights.bin", FILE_READ);
  if (!f) return false;
  f.read((uint8_t*)myConv1_w, CONV1_WEIGHTS*4);
  f.read((uint8_t*)myConv1_b, CONV1_FILTERS*4);
  f.read((uint8_t*)myConv2_w, CONV2_WEIGHTS*4);
  f.read((uint8_t*)myConv2_b, CONV2_FILTERS*4);
  f.read((uint8_t*)myOutput_w, OUTPUT_WEIGHTS*4);
  f.read((uint8_t*)myOutput_b, NUM_CLASSES*4);
  f.close();
  Serial.println("ANN weights loaded from SD");
  return true;
}
bool myLoadSnnWeights() {
  if (!mySDavailable) { Serial.println("No SD card - skipping SNN weight load"); return false; }
  if (!SD.exists("/header/mySnnWeights.bin")) { Serial.println("No /header/mySnnWeights.bin found"); return false; }
  File f = SD.open("/header/mySnnWeights.bin", FILE_READ);
  if (!f) return false;
  f.read((uint8_t*)mySnnConv1_w, CONV1_WEIGHTS*4);
  f.read((uint8_t*)mySnnConv1_b, CONV1_FILTERS*4);
  f.read((uint8_t*)mySnnConv2_w, CONV2_WEIGHTS*4);
  f.read((uint8_t*)mySnnConv2_b, CONV2_FILTERS*4);
  f.read((uint8_t*)mySnnOutput_w, OUTPUT_WEIGHTS*4);
  f.read((uint8_t*)mySnnOutput_b, NUM_CLASSES*4);
  f.close();
  Serial.println("SNN weights loaded from SD");
  return true;
}

// ======================================================
// FORWARD DECLARATIONS
// ======================================================
void myActionCollect(int classIdx);
void myActionInferAnn();
void myActionInferSnn();
void myResetMenuState();
void myHandleMenuNavigation();
void myDrawMenu();
bool myCaptureAndPreprocess();
void myAnnForward();
void mySnnForwardInfer();

// ======================================================
// PART 0: SETUP AND LOOP
// ======================================================
void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000);
  delay(1000);
  Serial.println("\n=== XIAO ESP32-S3 Vision ML (ANN+SNN infer) Starting ===");
  Serial.printf("Free heap: %d bytes\n", ESP.getFreeHeap());
  Serial.printf("Free PSRAM: %d bytes\n", ESP.getFreePsram());

  myRgbBuffer = (uint8_t*)ps_malloc(240*240*3);
  if (!myRgbBuffer) Serial.println("Failed to allocate RGB buffer!");

  pinMode(A0, INPUT);
  u8g2.begin();

  pinMode(21, OUTPUT); digitalWrite(21, HIGH); delay(100);
  Serial.println("Checking SD card...");
  SPI.begin(); SPI.setFrequency(400000);
  mySDavailable = SD.begin(21, SPI, 400000, "/sd", 5, false);
  if (!mySDavailable) {
    SD.end();
    Serial.println("No SD card - continuing without it");
    u8g2.firstPage(); do { u8g2.drawStr(0, 15, "No SD card"); } while (u8g2.nextPage());
    delay(2000);
  } else {
    Serial.println("SD card mounted successfully");
  }

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0; config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM; config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM; config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM; config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM; config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM; config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM; config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM; config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM; config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000; config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_240X240; config.jpeg_quality = 12;
  config.fb_count = 1;
  esp_camera_init(&config);
  Serial.println("Camera initialized");

  sensor_t* s = esp_camera_sensor_get();
  if (s != NULL) s->set_hmirror(s, 1);

  esp_log_level_set("*", ESP_LOG_WARN);
  esp_log_level_set("esp_camera", ESP_LOG_ERROR);

  myAllocateMemory();

  #ifdef USE_BAKED_WEIGHTS
    memcpy(myConv1_w, myModel_conv1_w, CONV1_WEIGHTS * sizeof(float));
    memcpy(myConv1_b, myModel_conv1_b, CONV1_FILTERS * sizeof(float));
    memcpy(myConv2_w, myModel_conv2_w, CONV2_WEIGHTS * sizeof(float));
    memcpy(myConv2_b, myModel_conv2_b, CONV2_FILTERS * sizeof(float));
    memcpy(myOutput_w, myModel_output_w, OUTPUT_WEIGHTS * sizeof(float));
    memcpy(myOutput_b, myModel_output_b, NUM_CLASSES * sizeof(float));
    myAnnTrained = true;
    memcpy(mySnnConv1_w, mySnnModel_conv1_w, CONV1_WEIGHTS * sizeof(float));
    memcpy(mySnnConv1_b, mySnnModel_conv1_b, CONV1_FILTERS * sizeof(float));
    memcpy(mySnnConv2_w, mySnnModel_conv2_w, CONV2_WEIGHTS * sizeof(float));
    memcpy(mySnnConv2_b, mySnnModel_conv2_b, CONV2_FILTERS * sizeof(float));
    memcpy(mySnnOutput_w, mySnnModel_output_w, OUTPUT_WEIGHTS * sizeof(float));
    memcpy(mySnnOutput_b, mySnnModel_output_b, NUM_CLASSES * sizeof(float));
    mySnnTrained = true;
    Serial.println("Baked-in weights loaded (both models)");
  #endif

  if (myLoadAnnWeights()) { Serial.println("SD ANN weights override baked-in"); myAnnTrained = true; }
  if (myLoadSnnWeights()) { Serial.println("SD SNN weights override baked-in"); mySnnTrained = true; }

  myLastActivityTime = millis();
  myResetMenuState();
  delay(2000);
  Serial.println("System ready - Tap A0 to navigate, 3+ taps to select");
  myDrawMenu();
}

void loop() { myHandleMenuNavigation(); }

// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 1: IMAGE COLLECTION FUNCTIONS (unchanged behaviour from v44)      ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████
void myRenderRgbToOLED(int imageCount) {
  int myOledWidth = u8g2.getDisplayWidth();
  int myOledHeight = u8g2.getDisplayHeight();
  int myScaleX = 240 / myOledWidth;
  int myScaleY = 240 / myOledHeight;
  u8g2.firstPage();
  do {
    for (int myOledX = 0; myOledX < myOledWidth; myOledX++) {
      for (int myOledY = 0; myOledY < myOledHeight; myOledY++) {
        size_t myPixelIndex = ((myOledY * myScaleY) * 240 + (myOledX * myScaleX)) * 3;
        uint8_t myBrightness = (myRgbBuffer[myPixelIndex] + myRgbBuffer[myPixelIndex+1] + myRgbBuffer[myPixelIndex+2]) / 3;
        if (myBrightness > 100) u8g2.drawPixel(myOledX, myOledY);
      }
    }
    if (imageCount >= 0) {
      u8g2.setFont(u8g2_font_ncenB10_tr);
      u8g2.setColorIndex(0); u8g2.drawBox(0, 0, 20, 15);
      u8g2.setColorIndex(1); u8g2.setCursor(3, 10); u8g2.print(String(imageCount));
    } else {
      u8g2.setFont(u8g2_font_5x7_tf);
      u8g2.setColorIndex(0); u8g2.drawBox(50, 0, 22, 8);
      u8g2.setColorIndex(1); u8g2.drawStr(52, 7, "LIVE");
    }
  } while (u8g2.nextPage());
}
void myDisplayImageOnOLED(camera_fb_t* fb, int imageCount) {
  if (!myRgbBuffer) { Serial.println("RGB buffer not allocated - skipping OLED preview"); return; }
  if (!fmt2rgb888(fb->buf, fb->len, fb->format, myRgbBuffer)) { Serial.println("Failed to convert JPEG to RGB888 for OLED"); return; }
  myRenderRgbToOLED(imageCount);
}
void myActionCollect(int classIdx) {
  if (!mySDavailable) {
    Serial.println("No SD card - cannot collect images");
    u8g2.firstPage(); do { u8g2.drawStr(0, 15, "No SD card"); } while (u8g2.nextPage());
    delay(2000); myResetMenuState(); return;
  }
  Serial.printf("\n>>> Collection mode: %s\n", myClassLabels[classIdx].c_str());
  Serial.println("  TAP (1-2 taps) = Capture image");
  Serial.println("  LONG PRESS (3+ taps) = Exit to menu");
  Serial.println("  Serial: 'T'=capture, 'L'=exit");
  myResetTouchState();

  String path = "/images/" + myClassLabels[classIdx];
  if (!SD.exists("/images")) SD.mkdir("/images");
  if (!SD.exists(path)) SD.mkdir(path);

  int counts[NUM_CLASSES] = {};
  File root = SD.open("/images/" + myClassLabels[classIdx]);
  if (root) {
    while (File file = root.openNextFile()) {
      if (!file.isDirectory() && (String(file.name()).endsWith(".jpg") || String(file.name()).endsWith(".JPG"))) counts[classIdx]++;
      file.close();
    }
    root.close();
  }

  unsigned long lastCameraDrain = 0, lastOLED = 0;
  bool oledNeedsUpdate = false, shouldCapture = false;
  while (true) {
    unsigned long now = millis();
    if (now - lastCameraDrain > 50) {
      lastCameraDrain = now;
      if (!shouldCapture) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (fb) {
          if (now - lastOLED > 250 && myRgbBuffer) {
            if (fmt2rgb888(fb->buf, fb->len, fb->format, myRgbBuffer)) { oledNeedsUpdate = true; lastOLED = now; }
          }
          esp_camera_fb_return(fb);
        }
      }
    }
    if (oledNeedsUpdate) { oledNeedsUpdate = false; myRenderRgbToOLED(-1); }

    if (Serial.available()) {
      char c = Serial.read();
      if (c == 'l' || c == 'L') { myResetMenuState(); return; }
      else if (c == 't' || c == 'T') shouldCapture = true;
    }
    int touchAction = myCheckTouchInput();
    if (touchAction == 2) { Serial.println("Exiting collection mode"); myResetMenuState(); return; }
    else if (touchAction == 1) shouldCapture = true;

    if (shouldCapture) {
      shouldCapture = false;
      camera_fb_t* fb = esp_camera_fb_get();
      if (fb) {
        String fileName = path + "/img_" + String(millis()) + ".jpg";
        File file = SD.open(fileName, FILE_WRITE);
        if (file) {
          file.write(fb->buf, fb->len); file.close();
          counts[classIdx]++;
          Serial.printf("Saved: %s (Total: %d)\n", fileName.c_str(), counts[classIdx]);
          myDisplayImageOnOLED(fb, counts[classIdx]);
          delay(300); lastOLED = millis();
        }
        esp_camera_fb_return(fb);
      }
    }
    delay(5);
  }
}

// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 2: FORWARD PASSES (ANN and SNN) - INFERENCE ONLY, no backward pass ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████

// Grabs one camera frame, decodes it, resizes to INPUT_SIZE, and fills
// myInputBuffer with normalized 0-1 values (NUM_CHANNELS per pixel; if
// NUM_CHANNELS==1 the RGB frame is luminance-converted here). Also leaves
// myRgbBuffer holding the full 240x240 RGB frame for the OLED preview.
// Shared by both myActionInferAnn() and myActionInferSnn().
bool myCaptureAndPreprocess() {
  static int sy_lookup[INPUT_SIZE], sx_lookup[INPUT_SIZE];
  static bool lookupInit = false;
  if (!lookupInit) {
    for (int i=0;i<INPUT_SIZE;i++) {
      sy_lookup[i] = min((int)((i+0.5)*240.0/INPUT_SIZE), 239);
      sx_lookup[i] = min((int)((i+0.5)*240.0/INPUT_SIZE), 239);
    }
    lookupInit = true;
  }
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) return false;
  bool ok = fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, myRgbBuffer);
  if (ok) {
    for (int y=0;y<INPUT_SIZE;y++) {
      int sy = sy_lookup[y];
      for (int x=0;x<INPUT_SIZE;x++) {
        int sx = sx_lookup[x];
        int srcIdx = (sy*240+sx)*3;
        int dstIdx = (y*INPUT_SIZE+x)*NUM_CHANNELS;
        #if NUM_CHANNELS == 1
          float r=myRgbBuffer[srcIdx], g=myRgbBuffer[srcIdx+1], b=myRgbBuffer[srcIdx+2];
          myInputBuffer[dstIdx] = (0.299f*r + 0.587f*g + 0.114f*b) * 0.003921569f;
        #else
          myInputBuffer[dstIdx]   = myRgbBuffer[srcIdx]   * 0.003921569f;
          myInputBuffer[dstIdx+1] = myRgbBuffer[srcIdx+1] * 0.003921569f;
          myInputBuffer[dstIdx+2] = myRgbBuffer[srcIdx+2] * 0.003921569f;
        #endif
      }
    }
  }
  esp_camera_fb_return(fb);
  return ok;
}

// ---- ANN forward pass: fills myDense_output[NUM_CLASSES] with softmax probabilities ----
void myAnnForward() {
  for (int f=0; f<CONV1_FILTERS; f++) {
    int ob = f*CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE;
    for (int y=0;y<CONV1_OUTPUT_SIZE;y++) for (int x=0;x<CONV1_OUTPUT_SIZE;x++) {
      float sum = myConv1_b[f];
      for (int ky=0;ky<CONV1_KERNEL_SIZE;ky++) for (int kx=0;kx<CONV1_KERNEL_SIZE;kx++) {
        int inPos = ((y+ky)*INPUT_SIZE+(x+kx))*NUM_CHANNELS;
        int wPos = f*CONV1_W_PER_FILTER + ky*CONV1_KERNEL_SIZE*NUM_CHANNELS + kx*NUM_CHANNELS;
        for (int c=0;c<NUM_CHANNELS;c++) sum += myInputBuffer[inPos+c]*myConv1_w[wPos+c];
      }
      myConv1_output[ob + y*CONV1_OUTPUT_SIZE + x] = leaky_relu(clip_value(sum));
    }
  }
  for (int f=0; f<CONV1_FILTERS; f++) {
    int ib=f*CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE, ob=f*POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE;
    for (int y=0;y<POOL1_OUTPUT_SIZE;y++) for (int x=0;x<POOL1_OUTPUT_SIZE;x++) {
      int iy=y*2, ix=x*2;
      float mv = myConv1_output[ib+iy*CONV1_OUTPUT_SIZE+ix];
      mv = max(mv, myConv1_output[ib+iy*CONV1_OUTPUT_SIZE+ix+1]);
      mv = max(mv, myConv1_output[ib+(iy+1)*CONV1_OUTPUT_SIZE+ix]);
      mv = max(mv, myConv1_output[ib+(iy+1)*CONV1_OUTPUT_SIZE+ix+1]);
      myPool1_output[ob+y*POOL1_OUTPUT_SIZE+x] = mv;
    }
  }
  for (int f=0; f<CONV2_FILTERS; f++) {
    int ob=f*CONV2_OUTPUT_SIZE*CONV2_OUTPUT_SIZE;
    for (int y=0;y<CONV2_OUTPUT_SIZE;y++) for (int x=0;x<CONV2_OUTPUT_SIZE;x++) {
      float sum = myConv2_b[f];
      for (int c=0;c<CONV1_FILTERS;c++) {
        int ib=c*POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE;
        for (int ky=0;ky<CONV2_KERNEL_SIZE;ky++) for (int kx=0;kx<CONV2_KERNEL_SIZE;kx++)
          sum += myPool1_output[ib+(y+ky)*POOL1_OUTPUT_SIZE+(x+kx)] * myConv2_w[f*CONV2_W_PER_FILTER+c*9+ky*3+kx];
      }
      myConv2_output[ob+y*CONV2_OUTPUT_SIZE+x] = leaky_relu(clip_value(sum));
    }
  }
  for (int c=0;c<NUM_CLASSES;c++) {
    double sum=0, comp=0;
    for (int i=0;i<FLATTENED_SIZE;i++) {
      double term = myConv2_output[i]*myOutput_w[c*FLATTENED_SIZE+i];
      double yv = term-comp, t = sum+yv;
      comp = (t-sum)-yv; sum = t;
    }
    myDense_output[c] = clip_value((float)sum + myOutput_b[c], -50, 50);
  }
  float mx = myDense_output[0]; for (int i=1;i<NUM_CLASSES;i++) mx = max(mx, myDense_output[i]);
  float expSum=0; for (int i=0;i<NUM_CLASSES;i++) expSum += exp(myDense_output[i]-mx);
  for (int i=0;i<NUM_CLASSES;i++) myDense_output[i] = exp(myDense_output[i]-mx)/expSum;
}

// ---- SNN forward pass (inference only): fills mySnnDense_output[NUM_CLASSES] ----
// Runs SNN_TIMESTEPS synthetic steps. Each step rate-codes myInputBuffer into a
// fresh Bernoulli spike frame, runs it through two LIF conv layers (membrane
// potential persists across steps), and accumulates a leaky per-class trace
// from the flattened Conv2 spikes. After the last step the trace is softmaxed.
void mySnnForwardInfer() {
  memset(mySnnC1Mem, 0, CONV1_FILTERS*CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE*sizeof(float));
  memset(mySnnC2Mem, 0, CONV2_FILTERS*CONV2_OUTPUT_SIZE*CONV2_OUTPUT_SIZE*sizeof(float));
  memset(mySnnTrace, 0, NUM_CLASSES*sizeof(float));

  for (int t=0; t<SNN_TIMESTEPS; t++) {
    // 1) rate-code this step's spike frame from the (0-1) input probabilities
    for (int i=0;i<INPUT_SIZE*INPUT_SIZE*NUM_CHANNELS;i++)
      mySnnInputSpike[i] = (((float)random(0,10001))/10000.0f < myInputBuffer[i]) ? 1 : 0;

    // 2) Conv1 LIF - instantaneous spatial conv each step (no lookback window;
    //    unlike the motion SNN, there's no real temporal pattern to extract here)
    for (int f=0; f<CONV1_FILTERS; f++) {
      int ob = f*CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE;
      for (int y=0;y<CONV1_OUTPUT_SIZE;y++) for (int x=0;x<CONV1_OUTPUT_SIZE;x++) {
        float current = mySnnConv1_b[f];
        for (int ky=0;ky<CONV1_KERNEL_SIZE;ky++) for (int kx=0;kx<CONV1_KERNEL_SIZE;kx++) {
          int inPos = ((y+ky)*INPUT_SIZE+(x+kx))*NUM_CHANNELS;
          int wPos = f*CONV1_W_PER_FILTER + ky*CONV1_KERNEL_SIZE*NUM_CHANNELS + kx*NUM_CHANNELS;
          for (int c=0;c<NUM_CHANNELS;c++) current += mySnnInputSpike[inPos+c]*mySnnConv1_w[wPos+c];
        }
        int idx = ob + y*CONV1_OUTPUT_SIZE + x;
        float memPre = mySnnC1Mem[idx]*LIF_LEAK + current;
        uint8_t spike = memPre >= LIF_THRESHOLD ? 1 : 0;
        mySnnC1Mem[idx] = memPre - (spike ? LIF_THRESHOLD : 0);
        mySnnC1Spike[idx] = spike;
      }
    }

    // 3) Pool1: 2x2 OR of binary spikes
    for (int f=0; f<CONV1_FILTERS; f++) {
      int ib=f*CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE, ob=f*POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE;
      for (int y=0;y<POOL1_OUTPUT_SIZE;y++) for (int x=0;x<POOL1_OUTPUT_SIZE;x++) {
        int iy=y*2, ix=x*2;
        uint8_t sOr = mySnnC1Spike[ib+iy*CONV1_OUTPUT_SIZE+ix] |
                      mySnnC1Spike[ib+iy*CONV1_OUTPUT_SIZE+ix+1] |
                      mySnnC1Spike[ib+(iy+1)*CONV1_OUTPUT_SIZE+ix] |
                      mySnnC1Spike[ib+(iy+1)*CONV1_OUTPUT_SIZE+ix+1];
        mySnnPooled[ob+y*POOL1_OUTPUT_SIZE+x] = sOr;
      }
    }

    // 4) Conv2 LIF
    for (int f=0; f<CONV2_FILTERS; f++) {
      int ob = f*CONV2_OUTPUT_SIZE*CONV2_OUTPUT_SIZE;
      for (int y=0;y<CONV2_OUTPUT_SIZE;y++) for (int x=0;x<CONV2_OUTPUT_SIZE;x++) {
        float current = mySnnConv2_b[f];
        for (int c=0;c<CONV1_FILTERS;c++) {
          int ib=c*POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE;
          for (int ky=0;ky<CONV2_KERNEL_SIZE;ky++) for (int kx=0;kx<CONV2_KERNEL_SIZE;kx++)
            current += mySnnPooled[ib+(y+ky)*POOL1_OUTPUT_SIZE+(x+kx)] * mySnnConv2_w[f*CONV2_W_PER_FILTER+c*9+ky*3+kx];
        }
        int idx = ob + y*CONV2_OUTPUT_SIZE + x;
        float memPre = mySnnC2Mem[idx]*LIF_LEAK + current;
        uint8_t spike = memPre >= LIF_THRESHOLD ? 1 : 0;
        mySnnC2Mem[idx] = memPre - (spike ? LIF_THRESHOLD : 0);
        mySnnC2Spike[idx] = spike;
      }
    }

    // 5) Output: flatten Conv2 spikes -> per-class current -> leaky trace
    for (int c=0;c<NUM_CLASSES;c++) {
      float current = mySnnOutput_b[c];
      for (int i=0;i<FLATTENED_SIZE;i++) current += mySnnC2Spike[i]*mySnnOutput_w[c*FLATTENED_SIZE+i];
      mySnnTrace[c] = mySnnTrace[c]*SNN_TRACE_LEAK + current;
    }
  }

  float mx = mySnnTrace[0]; for (int i=1;i<NUM_CLASSES;i++) mx = max(mx, mySnnTrace[i]);
  float expSum=0; for (int i=0;i<NUM_CLASSES;i++) expSum += exp(mySnnTrace[i]-mx);
  for (int i=0;i<NUM_CLASSES;i++) mySnnDense_output[i] = exp(mySnnTrace[i]-mx)/expSum;
}

// Shared OLED+Serial display for a completed inference frame - used by both infer modes.
void myShowInferenceResult(const char* modelTag, float* probs, unsigned long frameMs) {
  int pred = 0;
  for (int i=1;i<NUM_CLASSES;i++) if (probs[i] > probs[pred]) pred = i;
  int oW = u8g2.getDisplayWidth(), oH = u8g2.getDisplayHeight();
  int scX = 240/oW, scY = 240/oH;
  u8g2.firstPage();
  do {
    for (int ox=0; ox<oW; ox++) for (int oy=0; oy<oH; oy++) {
      int pi = ((oy*scY)*240 + (ox*scX))*3;
      uint8_t bright = (myRgbBuffer[pi]+myRgbBuffer[pi+1]+myRgbBuffer[pi+2])/3;
      if (bright > 100) u8g2.drawPixel(ox, oy);
    }
    u8g2.setFont(u8g2_font_5x7_tf);
    u8g2.setColorIndex(0); u8g2.drawBox(0, oH-9, oW, 9);
    u8g2.setColorIndex(1);
    char buf[24];
    snprintf(buf, sizeof(buf), "%s %d%%", myClassLabels[pred].c_str(), (int)(probs[pred]*100));
    u8g2.drawStr(1, oH-1, buf);
  } while (u8g2.nextPage());
  Serial.printf("[%s] %.1fms Pred: %s (%.1f%%) | All:", modelTag, (float)frameMs, myClassLabels[pred].c_str(), probs[pred]*100);
  for (int i=0;i<NUM_CLASSES;i++) Serial.printf(" %.0f%%", probs[i]*100);
  Serial.println();
}

// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 3: INFERENCE ACTIONS                                              ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████
void myActionInferAnn() {
  if (!myAnnTrained) {
    Serial.println("ERROR: No ANN weights loaded. Train in the browser and copy myWeights.bin to /header/ on the SD card.");
    u8g2.firstPage(); do { u8g2.drawStr(0,12,"No ANN weights"); u8g2.drawStr(0,24,"See browser"); } while (u8g2.nextPage());
    delay(2500); myResetMenuState(); return;
  }
  Serial.println("\n>>> Infer ANN. Serial/touch: t or l = exit");
  myResetTouchState();
  int frameCount = 0;
  while (true) {
    if (Serial.available()) { char c = Serial.read(); if (c=='t'||c=='T'||c=='l'||c=='L') { myResetMenuState(); return; } }
    unsigned long t0 = millis();
    if (myCaptureAndPreprocess()) {
      myAnnForward();
      myShowInferenceResult("ANN", myDense_output, millis()-t0);
    }
    frameCount++;
    if (frameCount >= 5) {
      frameCount = 0;
      if (myReadTouch() > myThresholdPress) { delay(200); myResetMenuState(); return; }
    }
  }
}
void myActionInferSnn() {
  if (!mySnnTrained) {
    Serial.println("ERROR: No SNN weights loaded. Train in the browser and copy mySnnWeights.bin to /header/ on the SD card.");
    u8g2.firstPage(); do { u8g2.drawStr(0,12,"No SNN weights"); u8g2.drawStr(0,24,"See browser"); } while (u8g2.nextPage());
    delay(2500); myResetMenuState(); return;
  }
  Serial.println("\n>>> Infer SNN. Serial/touch: t or l = exit");
  Serial.printf("    SNN_TIMESTEPS=%d LIF_LEAK=%.2f LIF_THRESHOLD=%.2f SNN_TRACE_LEAK=%.2f - must match the browser's training settings.\n",
                SNN_TIMESTEPS, LIF_LEAK, LIF_THRESHOLD, SNN_TRACE_LEAK);
  myResetTouchState();
  int frameCount = 0;
  while (true) {
    if (Serial.available()) { char c = Serial.read(); if (c=='t'||c=='T'||c=='l'||c=='L') { myResetMenuState(); return; } }
    unsigned long t0 = millis();
    if (myCaptureAndPreprocess()) {
      mySnnForwardInfer();
      myShowInferenceResult("SNN", mySnnDense_output, millis()-t0);
    }
    frameCount++;
    if (frameCount >= 5) {
      frameCount = 0;
      if (myReadTouch() > myThresholdPress) { delay(200); myResetMenuState(); return; }
    }
  }
}

// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 4: MENU SYSTEM FUNCTIONS                                          ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████
void myResetMenuState() { myIsSelected = false; myResetTouchState(); myLastActivityTime = millis(); myDrawMenu(); }

void myDrawMenu() {
  Serial.println("\n=== MENU ===");
  for (int i = 1; i <= myTotalItems; i++) {
    String label = (i <= NUM_CLASSES) ? myClassLabels[i-1] : (i == NUM_CLASSES+1) ? "Infer ANN" : "Infer SNN";
    Serial.print(i == myMenuIndex ? " > " : "   ");
    Serial.printf("%d. %s\n", i, label.c_str());
  }
  Serial.println("Commands: t=next (tap)  l=select (longpress)");
  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_6x10_tf);
    u8g2.drawStr(0, 8, "TAP:Next HOLD:Ok");
    int myStartItem = (myMenuIndex <= NUM_CLASSES) ? 1 : myMenuIndex - 2;
    for (int i=0;i<3;i++) {
      int cur = myStartItem + i;
      if (cur > myTotalItems) break;
      String label = (cur <= NUM_CLASSES) ? myClassLabels[cur-1] : (cur == NUM_CLASSES+1) ? "Infer ANN" : "Infer SNN";
      int y = 18 + i*9;
      u8g2.drawStr(0, y, (cur == myMenuIndex ? ("> "+label) : ("  "+label)).c_str());
    }
  } while (u8g2.nextPage());
}
void myExecuteMenuItem(int idx) {
  if (idx <= NUM_CLASSES) myActionCollect(idx-1);
  else if (idx == NUM_CLASSES+1) myActionInferAnn();
  else myActionInferSnn();
}
void myHandleMenuNavigation() {
  unsigned long now = millis();
  if (!myIsSelected && Serial.available()) {
    char c = Serial.read();
    if (c >= '1' && c <= '9') {
      int newIndex = c - '0';
      if (newIndex <= myTotalItems) { myMenuIndex = newIndex; myIsSelected = true; myLastActivityTime = now; myExecuteMenuItem(myMenuIndex); }
    } else if (c == 't' || c == 'T') {
      if (now - myLastTapTime > myTapCooldown) { myMenuIndex++; if (myMenuIndex > myTotalItems) myMenuIndex = 1; myDrawMenu(); myLastTapTime = now; myLastActivityTime = now; }
    } else if (c == 'l' || c == 'L') {
      myIsSelected = true; myLastActivityTime = now; myExecuteMenuItem(myMenuIndex);
    }
  }
  if (!myIsSelected) {
    int touchAction = myCheckTouchInput();
    if (touchAction == 1) {
      if (now - myLastTapTime > myTapCooldown) { myMenuIndex++; if (myMenuIndex > myTotalItems) myMenuIndex = 1; myDrawMenu(); myLastTapTime = now; myLastActivityTime = now; }
    } else if (touchAction == 2) {
      myIsSelected = true; myLastActivityTime = now; myExecuteMenuItem(myMenuIndex);
    }
  }
}
