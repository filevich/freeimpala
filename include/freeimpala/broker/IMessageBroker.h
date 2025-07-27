#pragma once
#include <string>
#include <functional>

class IMessageBroker {
public:
    virtual ~IMessageBroker() = default;
    virtual bool publish(const std::string& topic, const std::string& message) = 0;
    virtual bool subscribe(const std::string& topic) = 0;
    virtual void setMessageHandler(std::function<void(const std::string& topic, const std::string& message)> handler) = 0;
    virtual void loop() = 0;
};
