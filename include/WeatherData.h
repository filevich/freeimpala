#pragma once
#include <string>
#include <sstream>
#include <stdexcept>

struct WeatherData {
    int temperature;
    std::string location;
    float wind;
};

// Simple serialization format: [temperature]|[location]|[wind]
inline std::string serializeWeatherData(const WeatherData& data) {
    return std::to_string(data.temperature) + "|" + 
           data.location + "|" + 
           std::to_string(data.wind);
}

inline WeatherData deserializeWeatherData(const std::string& str) {
    size_t pos1 = str.find('|');
    size_t pos2 = str.find('|', pos1 + 1);
    
    if (pos1 == std::string::npos || pos2 == std::string::npos) {
        throw std::runtime_error("Invalid weather data format");
    }
    
    return WeatherData{
        std::stoi(str.substr(0, pos1)),
        str.substr(pos1 + 1, pos2 - pos1 - 1),
        std::stof(str.substr(pos2 + 1))
    };
}