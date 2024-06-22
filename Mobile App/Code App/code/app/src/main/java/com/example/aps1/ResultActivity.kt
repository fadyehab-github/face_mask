package com.example.aps1

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageDecoder
import android.graphics.Matrix
import android.graphics.RectF
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.view.ViewTreeObserver
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.google.android.gms.vision.Frame
import com.google.android.gms.vision.face.FaceDetector
import kotlinx.android.synthetic.main.activity_result.resultImageView
import kotlinx.android.synthetic.main.activity_result.resultOverlayView
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import kotlin.math.min

class ResultActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_result)

        val viewTreeObserver = resultOverlayView.viewTreeObserver
        if (viewTreeObserver.isAlive) {
            viewTreeObserver.addOnGlobalLayoutListener(object :
                ViewTreeObserver.OnGlobalLayoutListener {
                override fun onGlobalLayout() {
                    resultOverlayView.viewTreeObserver.removeOnGlobalLayoutListener(this)
                    val overlayWidth = resultOverlayView.width
                    val overlayHeight = resultOverlayView.height
                    load(overlayWidth, overlayHeight)
                }
            })
        }
    }

    fun load(overlayWidth: Int, overlayHeight: Int) {
        val uri = intent.getStringExtra("uri")
        val selectedImageUri = Uri.parse(uri)

        val bitmap = if (Build.VERSION.SDK_INT < 28) {
            MediaStore.Images.Media.getBitmap(this.contentResolver, selectedImageUri)
        } else {
            val source = ImageDecoder.createSource(this.contentResolver, selectedImageUri)
            ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                decoder.setTargetSampleSize(1)
                decoder.isMutableRequired = true
            }
        }
        val faceDetector = FaceDetector.Builder(this).setTrackingEnabled(true).build()
        if (!faceDetector.isOperational) {
            AlertDialog.Builder(this)
                .setMessage("Could not set up the face detector!")
                .show()
            return
        }
        val transformedBitmap = transformBitmap(bitmap, overlayWidth, overlayHeight)
        val boundingBoxes = processBitmap(transformedBitmap, faceDetector)
        resultImageView.setImageBitmap(transformedBitmap)
        resultOverlayView.boundingBox = boundingBoxes
        resultOverlayView.invalidate()
    }

    private fun transformBitmap(bitmap1: Bitmap, overlayWidth: Int, overlayHeight: Int): Bitmap {
        val matrix = Matrix()
        var bitmap = bitmap1

        // Ensure bitmap is in ARGB_8888 format
        bitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)

        // Calculate the scale factor
        val scaleWidth = overlayWidth.toFloat() / bitmap.width
        val scaleHeight = overlayHeight.toFloat() / bitmap.height
        val scaleFactor = minOf(scaleWidth, scaleHeight)

        // Calculate new width and height based on the scale factor
        val newWidth = (bitmap.width * scaleFactor).toInt()
        val newHeight = (bitmap.height * scaleFactor).toInt()

        // Create a scaled bitmap
        bitmap = Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)

        // Create a new bitmap with the desired overlay size and black background
        val finalBitmap = Bitmap.createBitmap(overlayWidth, overlayHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(finalBitmap)
        canvas.drawColor(Color.BLACK) // Fill the background with black

        // Calculate the position to center the scaled bitmap
        val left = (overlayWidth - newWidth) / 2
        val top = (overlayHeight - newHeight) / 2

        // Draw the scaled bitmap onto the canvas
        canvas.drawBitmap(bitmap, left.toFloat(), top.toFloat(), null)

        return finalBitmap
    }

    private fun processBitmap(bitmap: Bitmap, faceDetector: FaceDetector): MutableList<Box> {
        val boundingBoxList = mutableListOf<Box>()
        val frame = Frame.Builder().setBitmap(bitmap).build()
        val faces = faceDetector.detect(frame)

        for (i in 0 until faces.size()) {
            val thisFace = faces.valueAt(i)
            val left = thisFace.position.x
            val top = thisFace.position.y
            val right = left + thisFace.width
            val bottom = top + thisFace.height
            val croppedBitmap = Bitmap.createBitmap(
                bitmap,
                left.toInt(),
                top.toInt(),
                min(thisFace.width.toInt(), bitmap.width - left.toInt()),
                min(thisFace.height.toInt(), bitmap.height - top.toInt())
            )
            val label = predict(croppedBitmap)

            val withMaskProbability = label["WithMask"] ?: 0F
            val withoutMaskProbability = label["WithoutMask"] ?: 0F
            val IncorrectProbability = label["Incorrect"] ?: 0F
            var predictions = ""
            var kind =0;
            if (withMaskProbability > withoutMaskProbability && withMaskProbability > IncorrectProbability) {
                predictions = "With Mask: ${String.format("%.1f", withMaskProbability * 100)}%"
                kind = 0;
            } else if (withoutMaskProbability > withMaskProbability && withoutMaskProbability > IncorrectProbability) {
                predictions =
                    "Without Mask: ${String.format("%.1f", withoutMaskProbability * 100)}%"
                kind = 1
            } else if (IncorrectProbability > withoutMaskProbability && IncorrectProbability > withMaskProbability) {
                predictions =
                    "Incorrect Mask: ${String.format("%.1f", IncorrectProbability * 100)}%"
                kind = 2
            }


            boundingBoxList.add(
                Box(
                    RectF(left, top, right, bottom),
                    predictions,
                    kind
                )
            )
        }
        return boundingBoxList
    }

    private fun predict(input: Bitmap): MutableMap<String, Float> {
        val modelFile = FileUtil.loadMappedFile(this, "model.tflite")
        val model = Interpreter(modelFile, Interpreter.Options())
        val labels = FileUtil.loadLabels(this, "labels.txt")

        val imageDataType = model.getInputTensor(0).dataType()
        val inputShape = model.getInputTensor(0).shape()

        val outputDataType = model.getOutputTensor(0).dataType()
        val outputShape = model.getOutputTensor(0).shape()

        var inputImageBuffer = TensorImage(imageDataType)
        val outputBuffer = TensorBuffer.createFixedSize(outputShape, outputDataType)

        val cropSize = min(input.width, input.height)
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeWithCropOrPadOp(cropSize, cropSize))
            .add(ResizeOp(inputShape[1], inputShape[2], ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
            .add(NormalizeOp(255f, 255f))
            .build()

        inputImageBuffer.load(input)
        inputImageBuffer = imageProcessor.process(inputImageBuffer)

        model.run(inputImageBuffer.buffer, outputBuffer.buffer.rewind())

        val labelOutput = TensorLabel(labels, outputBuffer)
        return labelOutput.mapWithFloatValue
    }
}
