#pragma once
#include <string>
#include <sstream>
#include <vector>

namespace signals {
    class SimpleSerializer {
    public:
        static std::string serialize(
            const std::vector<std::pair<std::string, std::string>>& fields
        ) {
            std::stringstream ss;
            for (size_t i = 0; i < fields.size(); ++i) {
                ss << fields[i].first << ":" << fields[i].second;
                if (i < fields.size() - 1) {
                    ss << "|";
                }
            }
            return ss.str();
        }
        
        static std::vector<std::pair<std::string, std::string>> deserialize(
            const std::string& data
        ) {
            std::vector<std::pair<std::string, std::string>> result;
            std::stringstream ss(data);
            std::string field;
            
            while (std::getline(ss, field, '|')) {
                size_t colonPos = field.find(':');
                if (colonPos != std::string::npos) {
                    std::string key = field.substr(0, colonPos);
                    std::string value = field.substr(colonPos + 1);
                    result.push_back({key, value});
                }
            }
            
            return result;
        }
        
        static std::string getValue(
            const std::vector<std::pair<std::string, std::string>>& fields, 
            const std::string& key
        ) {
            auto it = std::find_if(fields.begin(), fields.end(),
                [&key](const auto& pair) { return pair.first == key; });
            return (it != fields.end()) ? it->second : "";
        }
    };
}