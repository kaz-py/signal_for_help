#include "serial_comm.hpp"
#include <iostream>

// ============================================================================
// Windows implementation
// ============================================================================
#ifdef _WIN32
#include <windows.h>

SerialComm::SerialComm() : handle_(INVALID_HANDLE_VALUE), open_(false) {}

SerialComm::~SerialComm() { close(); }

bool SerialComm::open(const std::string& port, int baudRate) {
    // Ports above COM9 require the \\.\ prefix
    std::string fullPort = (port.rfind("\\\\.\\", 0) == 0) ? port : "\\\\.\\" + port;

    handle_ = CreateFileA(fullPort.c_str(),
                          GENERIC_WRITE,
                          0, nullptr,
                          OPEN_EXISTING,
                          FILE_ATTRIBUTE_NORMAL,
                          nullptr);

    if (handle_ == INVALID_HANDLE_VALUE) {
        std::cerr << "[Serial] No se pudo abrir " << port
                  << " (error " << GetLastError() << ")\n";
        return false;
    }

    DCB dcb{};
    dcb.DCBlength = sizeof(dcb);
    if (!GetCommState(handle_, &dcb)) {
        CloseHandle(handle_);
        handle_ = INVALID_HANDLE_VALUE;
        return false;
    }

    dcb.BaudRate = static_cast<DWORD>(baudRate);
    dcb.ByteSize = 8;
    dcb.StopBits = ONESTOPBIT;
    dcb.Parity   = NOPARITY;

    if (!SetCommState(handle_, &dcb)) {
        CloseHandle(handle_);
        handle_ = INVALID_HANDLE_VALUE;
        return false;
    }

    COMMTIMEOUTS to{};
    to.WriteTotalTimeoutConstant   = 500;
    to.WriteTotalTimeoutMultiplier = 10;
    SetCommTimeouts(handle_, &to);

    portName_ = port;
    open_     = true;
    std::cout << "[Serial] Puerto " << port << " abierto a " << baudRate << " bps\n";
    return true;
}

void SerialComm::close() {
    if (open_ && handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(handle_);
        handle_ = INVALID_HANDLE_VALUE;
        open_   = false;
        std::cout << "[Serial] Puerto " << portName_ << " cerrado\n";
    }
}

bool SerialComm::send(const std::string& message) {
    if (!open_) return false;
    DWORD written = 0;
    BOOL  ok = WriteFile(handle_,
                         message.c_str(),
                         static_cast<DWORD>(message.size()),
                         &written, nullptr);
    if (!ok || written != static_cast<DWORD>(message.size())) {
        std::cerr << "[Serial] Error al escribir (error " << GetLastError() << ")\n";
        return false;
    }
    return true;
}

// ============================================================================
// POSIX implementation (Linux / macOS)
// ============================================================================
#else
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>

SerialComm::SerialComm() : fd_(-1), open_(false) {}

SerialComm::~SerialComm() { close(); }

static speed_t baudToSpeed(int baud) {
    switch (baud) {
        case 4800:   return B4800;
        case 9600:   return B9600;
        case 19200:  return B19200;
        case 38400:  return B38400;
        case 57600:  return B57600;
        case 115200: return B115200;
        default:     return B9600;
    }
}

bool SerialComm::open(const std::string& port, int baudRate) {
    fd_ = ::open(port.c_str(), O_WRONLY | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) {
        std::cerr << "[Serial] No se pudo abrir " << port
                  << ": " << std::strerror(errno) << "\n";
        return false;
    }

    // Switch to blocking after open (O_NONBLOCK only to bypass DCD line)
    int flags = fcntl(fd_, F_GETFL, 0);
    fcntl(fd_, F_SETFL, flags & ~O_NONBLOCK);

    struct termios tty{};
    if (tcgetattr(fd_, &tty) < 0) {
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    speed_t speed = baudToSpeed(baudRate);
    cfsetospeed(&tty, speed);
    cfsetispeed(&tty, speed);

    // 8N1, no flow control
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |=  CS8;
    tty.c_cflag &= ~CRTSCTS;
    tty.c_cflag |=  CREAD | CLOCAL;

    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    tty.c_iflag &= ~(IXON | IXOFF | IXANY | ICRNL);
    tty.c_oflag &= ~OPOST;

    tty.c_cc[VMIN]  = 0;
    tty.c_cc[VTIME] = 10;  // 1 s write timeout

    if (tcsetattr(fd_, TCSANOW, &tty) < 0) {
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    portName_ = port;
    open_     = true;
    std::cout << "[Serial] Puerto " << port << " abierto a " << baudRate << " bps\n";
    return true;
}

void SerialComm::close() {
    if (open_ && fd_ >= 0) {
        ::close(fd_);
        fd_   = -1;
        open_ = false;
        std::cout << "[Serial] Puerto " << portName_ << " cerrado\n";
    }
}

bool SerialComm::send(const std::string& message) {
    if (!open_) return false;
    ssize_t written = ::write(fd_, message.c_str(), message.size());
    if (written < 0 || static_cast<size_t>(written) != message.size()) {
        std::cerr << "[Serial] Error al escribir: " << std::strerror(errno) << "\n";
        return false;
    }
    return true;
}

#endif
