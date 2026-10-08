#Raspberry Pi Differential Drive Robot

A Python-controlled differential-drive robot powered by a Raspberry Pi and an L9110S motor driver, featuring proportional steering via USB/Bluetooth gamepad and headless startup.

## Wiring Diagram

| L9110S Pin | Raspberry Pi Pin | Function |
| :--- | :--- | :--- |
| **B-1A** | GPIO 17 | Right Motor Forward |
| **B-1B** | GPIO 27 | Right Motor Backward |
| **A-1A** | GPIO 22 | Left Motor Forward |
| **A-1B** | GPIO 23 | Left Motor Backward |
| **GND** | GND | Common Ground |
| **VCC** | Motor Battery (+) | External Motor Power |

> ⚠️ **Important:** Ensure the external motor battery ground connects directly to both the L9110S `GND` and a Raspberry Pi `GND` pin to maintain a stable logic level and prevent Pi brownouts.


### Controls
* **Left Stick (Y-Axis):** Proportional throttle (Forward / Backward)
* **Right Stick (X-Axis):** Proportional differential steering (Left / Right)
* **Ctrl + C:** Safe terminal shutdown (stops motors and releases GPIO pins)
