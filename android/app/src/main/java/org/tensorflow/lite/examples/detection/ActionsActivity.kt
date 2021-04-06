package org.tensorflow.lite.examples.detection

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity

class ActionsActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_actions)
        setSupportActionBar(findViewById(R.id.toolbar))

        findViewById<Button>(R.id.openCamera).setOnClickListener { view ->
            val intent = Intent(this, DetectorActivity::class.java)
            startActivity(intent)
        }

       /* findViewById<FloatingActionButton>(R.id.fab).setOnClickListener { view ->
            Snackbar.make(view, "Replace with your own action", Snackbar.LENGTH_LONG)
                    .setAction("Action", null).show()
        }*/
    }
}