package uz.inha.team26.stvt

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity
import uz.inha.team26.stvt.R


class ActionsActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_actions)
        setSupportActionBar(findViewById(R.id.toolbar))

        findViewById<Button>(R.id.openCamera).setOnClickListener { view ->
            val intent = Intent(this, DetectorActivity::class.java)
            startActivity(intent)
        }
    }



}