import serial
import collections
import numpy as np

class PPGDataCollector:
    """
    Handles the continuous PySerial connection to the microcontroller and manages
    the high-speed rolling data buffers for the Red and IR channels.
    """
    def __init__(self, port='COM12', baud_rate=115200, fs=100, window_seconds=60, slide_seconds=5):
        self.port = port
        self.baud_rate = baud_rate
        self.fs = fs
        self.window_size = window_seconds * fs
        self.slide_size = slide_seconds * fs
        
        # Using deque for O(1) time-complexity on append/pop operations
        # This prevents the CPU bottleneck that occurs when resizing standard Python lists
        self.red_buffer = collections.deque(maxlen=self.window_size)
        self.ir_buffer = collections.deque(maxlen=self.window_size)
        
        try:
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=1)
            self.ser.reset_input_buffer()
            print(f"✅ Successfully connected to hardware on {self.port} at {self.baud_rate} baud.")
        except Exception as e:
            print(f"❌ Hardware Connection Error on {self.port}: {e}")
            self.ser = None

    def start_streaming(self, processing_callback):
        """
        Listens to the serial port continuously. When the 60s buffer is full,
        it passes the data to the processing callback, then slides the window.
        """
        if not self.ser:
            print("Cannot start stream: Serial port not initialized.")
            return

        print(f"Buffering initial {self.window_size / self.fs} seconds of biometric data. Please hold still...")
        
        try:
            while True:
                if self.ser.in_waiting > 0:
                    try:
                        line = self.ser.readline().decode('utf-8').strip()
                        
                        if "Red:" in line and "IR:" in line:
                            parts = line.split(',')
                            red_val = int(parts[0].split(':')[1])
                            ir_val = int(parts[1].split(':')[1])
                            
                            self.red_buffer.append(red_val)
                            self.ir_buffer.append(ir_val)
                            
                            # Trigger processing when the 60-second window is completely full
                            if len(self.ir_buffer) == self.window_size:
                                
                                # Convert the fast deque to a standard numpy array for the math modules
                                red_np = np.array(self.red_buffer)
                                ir_np = np.array(self.ir_buffer)
                                
                                # Send data upward to the main execution script
                                processing_callback(red_np, ir_np)
                                
                                # Slide the window by dropping the oldest 5 seconds of data
                                for _ in range(self.slide_size):
                                    self.red_buffer.popleft()
                                    self.ir_buffer.popleft()
                                    
                    except (ValueError, IndexError, UnicodeDecodeError):
                        # Gracefully ignore corrupted bytes caused by transient voltage spikes
                        pass
                        
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Safely shuts down the serial port when exiting."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("\nSerial connection securely closed.")