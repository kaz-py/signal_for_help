#pragma once
#include <string>

// Cross-platform serial port writer (Linux POSIX / Windows Win32).
// Only write direction is needed — we send alerts to an external device.
class SerialComm {
public:
    SerialComm();
    ~SerialComm();

    // Open the port (e.g. "/dev/ttyUSB0" on Linux, "COM3" on Windows).
    // Returns true on success.
    bool open(const std::string& port, int baudRate = 9600);
    void close();

    bool        isOpen()   const { return open_; }
    std::string portName() const { return portName_; }

    // Write message to the port. Returns false on error.
    bool send(const std::string& message);

private:
#ifdef _WIN32
    void* handle_;   // HANDLE — avoids pulling <windows.h> into every TU
#else
    int fd_;
#endif
    std::string portName_;
    bool        open_{false};
};
