/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.tensorflow.lite.examples.detection

import android.graphics.*
import android.media.ImageReader.OnImageAvailableListener
import android.os.Bundle
import android.os.Environment
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.util.TypedValue
import android.view.View
import android.widget.Toast
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.core.Point
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.examples.detection.camera.CameraActivity
import org.tensorflow.lite.examples.detection.customview.OverlayView
import org.tensorflow.lite.examples.detection.env.BorderedText
import org.tensorflow.lite.examples.detection.env.ImageUtils
import org.tensorflow.lite.examples.detection.env.Logger
import org.tensorflow.lite.examples.detection.tflite.Classifier
import org.tensorflow.lite.examples.detection.tflite.Classifier.Recognition
import org.tensorflow.lite.examples.detection.tflite.YoloV4Classifier
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker
import java.io.File
import java.io.IOException
import java.util.*
import org.opencv.core.Size as OpencvCoreSize


/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
class DetectorActivity : CameraActivity(), OnImageAvailableListener {
    var trackingOverlay: OverlayView? = null
    private var sensorOrientation: Int = 0
    private var detector: Classifier? = null
    private var ocrDetector: Classifier? = null
    private var lastProcessingTimeMs: Long = 0
    private var rgbFrameBitmap: Bitmap? = null
    private var croppedBitmap: Bitmap? = null
    private var cropCopyBitmap: Bitmap? = null
    private var computingDetection = false
    private var timestamp: Long = 0
    private var frameToCropTransform: Matrix? = null;
    private var cropToFrameTransform: Matrix? = null;
    private var tracker: MultiBoxTracker? = null
    private var borderedText: BorderedText? = null


    val TESSBASE_PATH = Environment.getExternalStorageDirectory().toString()
    val DEFAULT_LANGUAGE = "eng"
    private val TESSDATA_PATH = "$TESSBASE_PATH/tessdata/"
    private val EXPECTED_CUBE_DATA_FILES_ENG = arrayOf(
            "eng.cube.bigrams",
            "eng.cube.fold",
            "eng.cube.lm",
            "eng.cube.nn",
            "eng.cube.params",
            "eng.cube.size",
            "eng.cube.word-freq",
            "eng.tesseract_cube.nn"
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState);
        LOGGER.i("OpenCV: " + OpenCVLoader.initDebug());
        val expectedFile = File(TESSDATA_PATH + File.separator +
                DEFAULT_LANGUAGE + ".traineddata")
        LOGGER.i("Tesseract lang file: " + expectedFile.exists())
    }

    public override fun onPreviewSizeChosen(size: Size, rotation: Int) {
        val textSizePx = TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, resources.displayMetrics)
        borderedText = BorderedText(textSizePx)
        borderedText!!.setTypeface(Typeface.MONOSPACE)
        tracker = MultiBoxTracker(this)
        var cropSize = TF_OD_API_INPUT_SIZE
        try {
            detector = YoloV4Classifier.create(
                    assets,
                    TF_OD_API_MODEL_FILE,
                    TF_OD_API_LABELS_FILE,
                    TF_OD_API_IS_QUANTIZED,
                    192, 540)
            cropSize = TF_OD_API_INPUT_SIZE
            ocrDetector = YoloV4Classifier.create(
                    assets,
                    TF_OD_API_OCR_MODEL_FILE,
                    TF_OD_API_OCR_LABELS_FILE,
                    TF_OD_API_IS_QUANTIZED,
                    TF_OD_API_OCR_INPUT_SIZE, 960)
        } catch (e: IOException) {
            e.printStackTrace()
            LOGGER.e(e, "Exception initializing classifier!")
            val toast = Toast.makeText(
                    applicationContext, "Classifier could not be initialized", Toast.LENGTH_SHORT)
            toast.show()
            finish()
        }
        previewWidth = size.width
        previewHeight = size.height
        sensorOrientation = rotation - screenOrientation
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation)
        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight)
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888)
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888)
        frameToCropTransform = ImageUtils.getTransformationMatrix(
                previewWidth, previewHeight,
                cropSize, cropSize,
                sensorOrientation!!, MAINTAIN_ASPECT)
        cropToFrameTransform = Matrix()
        frameToCropTransform?.invert(cropToFrameTransform)
        trackingOverlay = findViewById<View>(R.id.tracking_overlay) as OverlayView
        trackingOverlay!!.addCallback { canvas ->
            tracker!!.draw(canvas)
            if (isDebug) {
                tracker!!.drawDebug(canvas)
            }
        }
        tracker!!.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation!!)
    }

    override fun processImage() {
        ++timestamp
        val currTimestamp = timestamp
        trackingOverlay!!.postInvalidate()

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage()
            return
        }
        computingDetection = true
        //        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");
        rgbFrameBitmap!!.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight)
        readyForNextImage()
        val canvas = Canvas(croppedBitmap)
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null)
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap)
        }
        runInBackground { //                        LOGGER.i("Running detection on image " + currTimestamp);
            val startTime = SystemClock.uptimeMillis()
            val results = detector!!.recognizeImage(croppedBitmap)
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime
            Log.e("CHECK", "run: " + results.size)
            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap)
            val canvas = Canvas(cropCopyBitmap)
            val paint = Paint()
            paint.color = Color.RED
            paint.style = Paint.Style.STROKE
            paint.strokeWidth = 2.0f
            var minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API
            minimumConfidence = when (MODE) {
                DetectorMode.TF_OD_API -> MINIMUM_CONFIDENCE_TF_OD_API
            }
            val mappedRecognitions: MutableList<Recognition> = LinkedList()
            var lastLP = "";
            for (result in results) {
                val location = result.location
                if (location != null && result.confidence >= 0.65) {
                    canvas.drawRect(location, paint)
                    cropToFrameTransform!!.mapRect(location)
                    result.location = location
                    mappedRecognitions.add(result)

                    // Preparing image
                    var lpWidth = location.width();
                    var matrix = ImageUtils.getTransformationMatrix(
                            lpWidth.toInt(), location.height().toInt(),
                            TF_OD_API_OCR_INPUT_SIZE, Math.min(TF_OD_API_OCR_INPUT_SIZE, (TF_OD_API_OCR_INPUT_SIZE * location.height() / lpWidth).toInt()),
                            sensorOrientation, false)
                    var lpBitmap = Bitmap.createBitmap(rgbFrameBitmap,
                            location.left.toInt(), location.top.toInt(),
                            lpWidth.toInt(), location.height().toInt(),
                            matrix, false)


                    // OpenCV processing
                    var inputMat = Mat()
                    var greyMat = Mat()
                    var blurMat = Mat()
                    var processedMat = Mat()
                    Utils.bitmapToMat(lpBitmap, inputMat);
                    Imgproc.cvtColor(inputMat, greyMat, Imgproc.COLOR_RGB2GRAY)
                    Imgproc.GaussianBlur(greyMat, blurMat, OpencvCoreSize(3.0, 3.0), 2.5)
                    Imgproc.adaptiveThreshold(blurMat, processedMat, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 101, 6.0)
                    var finalImage : Bitmap = Bitmap.createBitmap(lpBitmap);
                    Utils.matToBitmap(blurMat, finalImage);

                    val ocrResults = ocrDetector?.recognizeImage(finalImage)
                    var resultStr = "";
                    ocrResults?.sortBy { it.location.left }
                    for (ocrResult in ocrResults!!)
                    {
                        val ocrLocation = ocrResult.getLocation()
                        if (ocrLocation != null && ocrResult.getConfidence() >= 0.6)
                        {
                            canvas.drawRect(ocrLocation, paint)
                            val invert = Matrix()
                            matrix.invert(invert)
                            invert.mapRect(ocrLocation)
                            ocrLocation.left += location.left
                            ocrLocation.right += location.left
                            ocrLocation.top += location.top
                            ocrLocation.bottom += location.top
                            ocrResult.setLocation(ocrLocation)
                            mappedRecognitions.add(ocrResult)
                        }
                        resultStr = "$resultStr${ocrResult.title}"
                    }
                    LOGGER.i("RESULT: %s", resultStr);
                    lastLP = resultStr;
                }
            }
            tracker!!.trackResults(mappedRecognitions, currTimestamp)
            trackingOverlay!!.postInvalidate()
            computingDetection = false
            runOnUiThread {
                showLPInfo(lastLP)
                showFrameInfo(previewWidth.toString() + "x" + previewHeight)
                showCropInfo(cropCopyBitmap?.getWidth().toString() + "x" + cropCopyBitmap?.getHeight())
                showInference(lastProcessingTimeMs.toString() + "ms")
            }
        }
    }

    override fun getLayoutId(): Int {
        return R.layout.tfe_od_camera_connection_fragment_tracking
    }

    override fun getDesiredPreviewFrameSize(): Size {
        return DESIRED_PREVIEW_SIZE
    }


    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.
    private enum class DetectorMode {
        TF_OD_API
    }

    override fun setUseNNAPI(isChecked: Boolean) {
        runInBackground { detector!!.setUseNNAPI(isChecked) }
        runInBackground { ocrDetector!!.setUseNNAPI(isChecked) }
    }

    override fun setNumThreads(numThreads: Int) {
        runInBackground { detector!!.setNumThreads(numThreads) }
        runInBackground { ocrDetector!!.setNumThreads(numThreads) }
    }

    companion object {
        private val LOGGER = Logger()
        private const val TF_OD_API_INPUT_SIZE = 192
        private const val TF_OD_API_OCR_INPUT_SIZE = 256
        private const val TF_OD_API_IS_QUANTIZED = true
        private const val TF_OD_API_MODEL_FILE = "yolov4-tiny_192_74maP_quantized.tflite"
        private const val TF_OD_API_LABELS_FILE = "file:///android_asset/coco.txt"
        private const val TF_OD_API_OCR_MODEL_FILE = "yolov4-tiny_ocr_256_100maP_quantized.tflite"
//        private const val TF_OD_API_OCR_MODEL_FILE = "yolov4-tiny_ocr_256_100maP_quantized.tflite"
        private const val TF_OD_API_OCR_LABELS_FILE = "file:///android_asset/ocr.txt"
        private val MODE = DetectorMode.TF_OD_API
        private const val MINIMUM_CONFIDENCE_TF_OD_API = 0.3f
        private const val MAINTAIN_ASPECT = true
        private val DESIRED_PREVIEW_SIZE = Size(640, 480)
        private const val SAVE_PREVIEW_BITMAP = false
        private const val TEXT_SIZE_DIP = 10f
    }
}