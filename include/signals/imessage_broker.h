// IMessageBroker.h
#pragma once
#include <string>
#include <functional>

namespace signals {
    /**
     * Interface for callback-based message brokers.
     * 
     * This interface assumes that the underlying message broker library
     * handles message processing automatically in background threads via callbacks.
     * As long as the main thread remains alive, the handler set via setMessageHandler()
     * will be invoked automatically for every incoming message.
     * 
     * Examples of compatible technologies: MQTT (with callbacks), Kafka (librdkafka), etc.
     * 
     * For polling-based message brokers that require explicit message processing loops,
     * a separate interface should be created in the future if needed.
     */
    class IMessageBroker {
    public:
        virtual ~IMessageBroker() = default;
        
        /**
         * Publish a message to the specified topic.
         * @param topic The topic to publish to
         * @param message The message content
         * @return true if message was successfully published, false otherwise
         */
        virtual bool publish(const std::string& topic, const std::string& message) = 0;
        
        /**
         * Subscribe to a topic to receive messages.
         * @param topic The topic to subscribe to
         * @return true if subscription was successful, false otherwise
         */
        virtual bool subscribe(const std::string& topic) = 0;
        
        /**
         * Set the callback function that will be invoked automatically
         * when messages arrive on subscribed topics.
         * @param handler Callback function that receives topic and message
         */
        virtual void setMessageHandler(std::function<void(const std::string& topic, const std::string& message)> handler) = 0;
    };
}