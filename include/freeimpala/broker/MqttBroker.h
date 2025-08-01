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
        std::cerr << "Connection lost";
        if (cause) {
            std::cerr << ": " << cause;
        }
        std::cerr << std::endl;
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
        
        std::cout << "Attempting to connect to MQTT broker at " << serverAddress << "..." << std::endl;
        
        int rc = MQTTClient_connect(client, &conn_opts);
        if (rc != MQTTCLIENT_SUCCESS) {
            std::cerr << "Failed to connect to MQTT broker, return code: " << rc << std::endl;
            if (rc == MQTTCLIENT_DISCONNECTED) {
                std::cerr << "  (Broker not running or incorrect address/port?)" << std::endl;
            }
            return false;
        }
        
        connected = true;
        std::cout << "Successfully connected to MQTT broker." << std::endl;
        return true;
    }
    
    // Disconnect from the broker
    void disconnect() {
        if (connected) {
            std::cout << "Disconnecting from MQTT broker..." << std::endl;
            int rc = MQTTClient_disconnect(client, 10000); // 10 seconds timeout
            if (rc != MQTTCLIENT_SUCCESS) {
                std::cerr << "Failed to disconnect, return code: " << rc << std::endl;
            } else {
                std::cout << "Disconnected from MQTT broker." << std::endl;
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
            std::cerr << "Failed to publish message to topic '" << topic << "', return code: " << rc << std::endl;
            return false;
        }
        
        // Wait for delivery confirmation for QoS > 0
        if (qos > 0) {
            rc = MQTTClient_waitForCompletion(client, token, 1000L);
            if (rc != MQTTCLIENT_SUCCESS) {
                std::cerr << "Message delivery failed or timed out for topic '" << topic << "', return code: " << rc << std::endl;
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
        
        std::cout << "Subscribing to topic: " << topic << std::endl;
        
        int rc = MQTTClient_subscribe(client, topic.c_str(), qos);
        if (rc != MQTTCLIENT_SUCCESS) {
            std::cerr << "Failed to subscribe to topic '" << topic << "', return code: " << rc << std::endl;
            return false;
        }
        
        std::cout << "Successfully subscribed to topic: " << topic << std::endl;
        return true;
    }
    
    // Set the message handler
    void setMessageHandler(std::function<void(const std::string& topic, const std::string& message)> handler) override {
        messageHandler = handler;
    }
    
    // Process messages
    void loop(bool blocking = false, int timeoutMs = 1000) override {
        if (!connected) {
            return;
        }
        
        if (blocking) {
            // Blocking mode: wait for messages with timeout
            // The MQTT library handles message callbacks in background threads,
            // so we just need to block and let them work
            std::this_thread::sleep_for(std::chrono::milliseconds(timeoutMs));
        } else {
            // Non-blocking mode: process pending messages and return immediately
            MQTTClient_yield();
            
            // Small sleep to prevent busy waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
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
