import keyboard

def emulate_channel_8():
    """Emulates rc channel 8 signal for simulation.
        Press 'a' once to turn on. Press again to turn off."""
    
    switch_on = False

    while True:
        if keyboard.press('a'):
            switch_on = not switch_on
            print("a pressed")
            
emulate_channel_8()
    
