#pragma once
#include "IMessageBroker.h"
#include <MQTTClient.h>
#include <iostream>
#include <thread>
#include <chrono>

class MqttBroker : public IMessageBroker {
private:
    MQTTClient client;
    std::string serverAddress;
    std::string clientId;
    std::function<void(const std::string&, const std::string&)> messageHandler;
    bool connected;
    int qos;
    
    // Static callback function for incoming messages
    static int messageArrivedCallback(void* context, char* topicName, int topicLen, MQTTClient_message* message) {
        MqttBroker* broker = static_cast<MqttBroker*>(context);
        return broker->handleMessage(topicName, topicLen, message);
    }
    
    // Static callback for connection lost
    static void connectionLostCallback(void* context, char* cause) {
        MqttBroker* broker = static_cast<MqttBroker*>(context);
        broker->handleConnectionLost(cause);
    }
    
    // Handle incoming message
    int handleMessage(char* topicName, int topicLen, MQTTClient_message* message) {
        if (messageHandler) {
            std::string topic;
            if (topicLen > 0) {
                topic = std::string(topicName, topicLen);
            } else {
                topic = std::string(topicName);
            }
            
            std::string msg(static_cast<char*>(message->payload), message->payloadlen);
            messageHandler(topic, msg);
        }
        
        MQTTClient_freeMessage(&message);
        MQTTClient_free(topicName);
        return 1; // Return 1 to indicate successful processing
    }
    
    // Handle connection lost
    void handleConnectionLost(char* cause) {
        connected = false;
        spdlog::error("Connection lost {}", cause);
    }

public:
    // Constructor
    MqttBroker(const std::string& serverAddr, const std::string& clientId, int qos = 1) 
        : serverAddress(serverAddr), clientId(clientId), connected(false), qos(qos) {
        
        int rc = MQTTClient_create(&client, serverAddress.c_str(), clientId.c_str(),
                                   MQTTCLIENT_PERSISTENCE_NONE, NULL);
        if (rc != MQTTCLIENT_SUCCESS) {
            throw std::runtime_error("Failed to create MQTT client, return code: " + std::to_string(rc));
        }
        
        // Set callbacks
        MQTTClient_setCallbacks(client, this, connectionLostCallback, messageArrivedCallback, NULL);
    }
    
    // Destructor
    ~MqttBroker() {
        disconnect();
        MQTTClient_destroy(&client);
    }
    
    // Connect to the broker
    bool connect() {
        if (connected) {
            return true;
        }
        
        MQTTClient_connectOptions conn_opts = MQTTClient_connectOptions_initializer;
        conn_opts.cleansession = 1;
        
        spdlog::info("Attempting to connect to MQTT broker at {}", serverAddress);
        
        int rc = MQTTClient_connect(client, &conn_opts);
        if (rc != MQTTCLIENT_SUCCESS) {
            spdlog::error("Failed to connect to MQTT broker, return code: {}", rc);
            if (rc == MQTTCLIENT_DISCONNECTED) {
                spdlog::error("Broker not running or incorrect address/port");
            }
            return false;
        }
        
        connected = true;
        spdlog::info("Successfully connected to MQTT broker");
        return true;
    }
    
    // Disconnect from the broker
    void disconnect() {
        if (connected) {
            spdlog::info("Disconnecting from MQTT broker");
            
            int rc = MQTTClient_disconnect(client, 10000); // 10 seconds timeout
            if (rc != MQTTCLIENT_SUCCESS) {
                spdlog::error("Failed to disconnect, return code: {}", rc);
            } else {
                spdlog::info("Disconnected from MQTT broker");
            }
            connected = false;
        }
    }
    
    // Publish a message
    bool publish(const std::string& topic, const std::string& message) override {
        if (!connected && !connect()) {
            return false;
        }
        
        MQTTClient_message pubmsg = MQTTClient_message_initializer;
        MQTTClient_deliveryToken token;
        
        pubmsg.payload = const_cast<char*>(message.c_str());
        pubmsg.payloadlen = message.length();
        pubmsg.qos = qos;
        pubmsg.retained = 0;
        
        int rc = MQTTClient_publishMessage(client, topic.c_str(), &pubmsg, &token);
        if (rc != MQTTCLIENT_SUCCESS) {
            spdlog::error("Failed to publish message to topic '{}', return code: {}", topic, rc);
            return false;
        }
        
        // Wait for delivery confirmation for QoS > 0
        if (qos > 0) {
            rc = MQTTClient_waitForCompletion(client, token, 1000L);
            if (rc != MQTTCLIENT_SUCCESS) {
                spdlog::error("Message delivery failed or timed out for topic '{}', return code: {}", topic, rc);
                return false;
            }
        }
        
        return true;
    }
    
    // Subscribe to a topic
    bool subscribe(const std::string& topic) override {
        if (!connected && !connect()) {
            return false;
        }
        
        spdlog::info("Subscribing to topic: {}", topic);
        
        int rc = MQTTClient_subscribe(client, topic.c_str(), qos);
        if (rc != MQTTCLIENT_SUCCESS) {
            spdlog::error("Failed to subscribe to topic '{}', return code: {}", topic, rc);
            return false;
        }
        
        spdlog::info("Successfully subscribed to topic: {}", topic);
        return true;
    }
    
    // Set the message handler
    void setMessageHandler(std::function<void(const std::string& topic, const std::string& message)> handler) override {
        messageHandler = handler;
    }
    
    // Process messages - for MQTT with callbacks, this is mostly a no-op
    // since message processing happens automatically in background threads
    void loop(bool blocking = false, int timeoutMs = 1000) {
        if (!connected) {
            return;
        }
        
        if (blocking) {
            // For MQTT with callbacks, we just need to keep the thread alive
            // Messages are processed automatically by the callback system
            std::this_thread::sleep_for(std::chrono::milliseconds(timeoutMs));
        } else {
            // Non-blocking: just yield briefly to let system process
            // any pending work, then return immediately
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    
    // Convenience methods
    bool isConnected() const {
        return connected;
    }
    
    void setQoS(int qos) {
        this->qos = qos;
    }
};