package com.example.aps1

import android.graphics.Canvas
import android.content.Context
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View


class OverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {
    var boundingBox: MutableList<Box> = mutableListOf()
    var paint = Paint()

    override fun draw(canvas: Canvas) {
        super.draw(canvas)
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 3f
        paint.strokeCap = Paint.Cap.ROUND
        paint.strokeJoin = Paint.Join.ROUND
        paint.strokeMiter = 100f
        boundingBox.forEach { box ->
            if (box.kind ==0){
                paint.color = Color.GREEN
            } else  if (box.kind ==1){
                paint.color = Color.RED
            }else{
                paint.color = Color.BLUE
            }
            paint.setTextAlign(Paint.Align.LEFT)
            paint.textSize = 54f // Set the text size to 24 (adjust size as needed)
            canvas.drawText(box.label, box.rect.left, box.rect.top-9F, paint)
            canvas.drawRoundRect(box.rect, 2F, 2F, paint)
        }
    }
}