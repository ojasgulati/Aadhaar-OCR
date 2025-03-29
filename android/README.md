## Usage

1. Adding Dependencies
```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite-task-vision-play-services:0.4.2'
    implementation 'com.google.android.gms:play-services-tflite-gpu:16.0.0'
    // To recognize Latin script
    implementation 'com.google.android.gms:play-services-mlkit-text-recognition:19.0.0'
}
```

2. Download model file from [here](https://github.com/ojasgulati/Aadhaar-OCR/blob/main/android/model-v0-noconfigs.tflite) (or create your own) and keep it in the assets folder.

3. Create Object Detection Helper Class
```kotlin
import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.google.android.gms.tflite.client.TfLiteInitializationOptions
import com.google.android.gms.tflite.gpu.support.TfLiteGpu
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.gms.vision.TfLiteVision
import org.tensorflow.lite.task.gms.vision.detector.Detection
import org.tensorflow.lite.task.gms.vision.detector.ObjectDetector

class ObjectDetectorHelper(
        var threshold: Float = 0.5f,
        var numThreads: Int = 2,
        var maxResults: Int = 3,
        var currentDelegate: Int = 0,
        var currentModel: Int = 0,
        val context: Context,
        val objectDetectorListener: DetectorListener
) {

    private val TAG = "ObjectDetectionHelper"

    // For this example this needs to be a var so it can be reset on changes. If the ObjectDetector
    // will not change, a lazy val would be preferable.
    private var objectDetector: ObjectDetector? = null
    private var gpuSupported = false

    init {

        TfLiteGpu.isGpuDelegateAvailable(context).onSuccessTask { gpuAvailable: Boolean ->
            val optionsBuilder =
                    TfLiteInitializationOptions.builder()
            if (gpuAvailable) {
                optionsBuilder.setEnableGpuDelegateSupport(true)
            }
            TfLiteVision.initialize(context, optionsBuilder.build())
        }.addOnSuccessListener {
            objectDetectorListener.onInitialized()
        }.addOnFailureListener {
            objectDetectorListener.onError("TfLiteVision failed to initialize: "
                    + it.message)
        }
    }

    fun clearObjectDetector() {
        objectDetector = null
    }

    // Initialize the object detector using current settings on the
    // thread that is using it. CPU and NNAPI delegates can be used with detectors
    // that are created on the main thread and used on a background thread, but
    // the GPU delegate needs to be used on the thread that initialized the detector
    fun setupObjectDetector() {
        if (!TfLiteVision.isInitialized()) {
            Log.e(TAG, "setupObjectDetector: TfLiteVision is not initialized yet")
            return
        }

        // Create the base options for the detector using specifies max results and score threshold
        val optionsBuilder =
                ObjectDetector.ObjectDetectorOptions.builder()
                        .setScoreThreshold(threshold)
                        .setMaxResults(maxResults)

        // Set general detection options, including number of used threads
        val baseOptionsBuilder = BaseOptions.builder().setNumThreads(numThreads)

        // Use the specified hardware for running the model. Default to CPU
        when (currentDelegate) {
            DELEGATE_CPU -> {
                // Default
            }

            DELEGATE_GPU -> {
                if (gpuSupported) {
                    baseOptionsBuilder.useGpu()
                } else {
                    objectDetectorListener.onError("GPU is not supported on this device")
                }
            }

            DELEGATE_NNAPI -> {
                baseOptionsBuilder.useNnapi()
            }
        }

        optionsBuilder.setBaseOptions(baseOptionsBuilder.build())

        val modelName = "model-v0-noconfigs.tflite" //todo keep mode file in assets folder

        try {
            objectDetector =
                    ObjectDetector.createFromFileAndOptions(context, modelName, optionsBuilder.build())
        } catch (e: Exception) {
            objectDetectorListener.onError(
                    "Object detector failed to initialize. See error logs for details"
            )
            Log.e(TAG, "TFLite failed to load model with error: " + e.message)
        }
    }

    fun detect(image: Bitmap, imageRotation: Int) {
        if (!TfLiteVision.isInitialized()) {
            Log.e(TAG, "detect: TfLiteVision is not initialized yet")
            return
        }

        if (objectDetector == null) {
            setupObjectDetector()
        }

        // Inference time is the difference between the system time at the start and finish of the
        // process
        var inferenceTime = SystemClock.uptimeMillis()

        // Create preprocessor for the image.
        // See https://www.tensorflow.org/lite/inference_with_metadata/
        //            lite_support#imageprocessor_architecture
        val imageProcessor = ImageProcessor.Builder().add(Rot90Op(-imageRotation / 90)).build()

        // Preprocess the image and convert it into a TensorImage for detection.
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))

        val results = objectDetector?.detect(tensorImage)
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime
        objectDetectorListener.onResults(
                bitmap = image,
                results,
                inferenceTime,
                tensorImage.height,
                tensorImage.width)
    }

    interface DetectorListener {
        fun onInitialized()
        fun onError(error: String)
        fun onResults(
                bitmap: Bitmap,
                results: MutableList<Detection>?,
                inferenceTime: Long,
                imageHeight: Int,
                imageWidth: Int
        )
    }

    companion object {
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DELEGATE_NNAPI = 2
        const val MODEL_MOBILENETV1 = 0
        const val MODEL_EFFICIENTDETV0 = 1
        const val MODEL_EFFICIENTDETV1 = 2
        const val MODEL_EFFICIENTDETV2 = 3
    }
}
```

4. After your image capture, pre process image and call the detect method of the ObjectDetectorHelper class
```kotlin
val bitmap = ImageDecoder.decodeBitmap(
                        ImageDecoder.createSource(
                            context.contentResolver,
                            uri
                        )
                    ).copy(Bitmap.Config.ARGB_8888, true)
                    val a = bitmap.toGrayscale()
                    objectDetectorHelper.detect(a, 0)

fun Bitmap.toGrayscale(): Bitmap {
    val bmpGrayscale = Bitmap.createBitmap(this.width, this.height, Bitmap.Config.ARGB_8888)
    val c = Canvas(bmpGrayscale)
    val paint = Paint()
    val cm = ColorMatrix()
    cm.setSaturation(0f)
    val f = ColorMatrixColorFilter(cm)
    paint.colorFilter = f
    c.drawBitmap(this, 0f, 0f, paint)
    val a = Bitmap.createScaledBitmap(bmpGrayscale, 640, 640, false)
    return a;
}
```

5. Implement the DetectorListener interface in your activity
```kotlin
for (result in results ?: emptyList()) {
            val boundingBox = result.boundingBox

            val top = boundingBox.top
            val bottom = boundingBox.bottom
            val left = boundingBox.left
            val right = boundingBox.right

            // Draw bounding box around detected objects
            val rect = RectF(left, top, right, bottom)
            val resultBmp = Bitmap.createBitmap(
                (rect.right - rect.left).toInt(),
                (rect.bottom - rect.top).toInt(),
                Bitmap.Config.ARGB_8888
            )
            Canvas(resultBmp).drawBitmap(bitmap, -rect.left, -rect.top, null);
            val image = InputImage.fromBitmap(resultBmp, 0)
            recognizer.process(image)
                .addOnSuccessListener { visionText ->
                    when(result.categories[0].label){
                        "dob"-> {
                            val dob = visionText.text.replace("-","").replace("/","")
                            print(dob)
                        }
                        "gender"->{
                            val gender = visionText.text.lowercase()
                            print(gender)
                        }
                        "name"->{
                            val name = visionText.text.split(" ")
                            print(name)
                        }
                        "id"->{
                            val id = visionText.text.replace(" ","")
                            print(id)
                        }
                    }
                }
                .addOnFailureListener { e ->
                }
        }
```

