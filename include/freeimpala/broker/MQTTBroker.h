#pragma once

#include "IMessageBroker.h"
#include "mqtt_c/mqtt.h"
#include <cstring>
#include <stdexcept>
#include <unistd.h>
#include <sys/socket.h>
#include <netdb.h>
#include <fcntl.h>
#include <thread>
#include <iostream>

class MQTTBroker : public IMessageBroker {
    int sockfd;
    struct mqtt_client client;
    uint8_t sendbuf[2048];
    uint8_t recvbuf[2048];
    std::thread worker;
    bool running;
    std::function<void(const std::string&, const std::string&)> msgHandler;
    
    static void publish_callback(void** state, struct mqtt_response_publish *msg) {
        MQTTBroker* broker = static_cast<MQTTBroker*>(*state);
        std::string topic((char*)msg->topic_name, msg->topic_name_size);
        std::string payload((char*)msg->application_message, msg->application_message_size);
        broker->msgHandler(topic, payload);
    }
    
    void connect(const char* host, int port) {
        struct hostent* hostent = gethostbyname(host);
        if (!hostent) throw std::runtime_error("DNS lookup failed");
        
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) throw std::runtime_error("Socket creation failed");
        
        struct sockaddr_in addr;
        std::memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        addr.sin_addr = *((struct in_addr*)hostent->h_addr);
        
        // CORRECTED connect call with proper parenthesis
        if (::connect(sockfd, (struct sockaddr*)&addr, sizeof(addr))) {
            close(sockfd);
            throw std::runtime_error("Connection failed");
        }
        
        // Initialize MQTT client with correct signature
        mqtt_init(&client, sockfd, sendbuf, sizeof(sendbuf), recvbuf, sizeof(recvbuf), 
                  publish_callback);
        
        // Set callback state
        client.publish_response_callback_state = this;
        
        // Send CONNECT
        const char* client_id = "mqtt_client_yup";
        uint8_t connect_flags = MQTT_CONNECT_CLEAN_SESSION;
        if (mqtt_connect(&client, client_id, nullptr, nullptr, 0, nullptr, nullptr, connect_flags, 400) != MQTT_OK) {
            close(sockfd);
            throw std::runtime_error("MQTT connect failed");
        }
        
        // Wait for CONNACK
        if (mqtt_sync(&client) != MQTT_OK) {
            close(sockfd);
            throw std::runtime_error("MQTT handshake failed");
        }
    }
    
    void worker_thread() {
        while (running) {
            if (mqtt_sync(&client) != MQTT_OK) {
                // Handle error? Maybe break?
                break;
            }
            usleep(100000); // 100ms
        }
    }
    
public:
    MQTTBroker(const char* host, int port) : running(true) {
        connect(host, port);
        worker = std::thread(&MQTTBroker::worker_thread, this);
    }
    
    ~MQTTBroker() {
        running = false;
        if (worker.joinable()) worker.join();
        close(sockfd);
    }
    
    bool publish(const std::string& topic, const std::string& message) override {
        uint8_t qos = 0;
        int ret = mqtt_publish(&client, topic.c_str(), message.data(), message.size(), qos);
        return ret == MQTT_OK;
    }
    
    bool subscribe(const std::string& topic) override {
        uint8_t qos = 0;
        int ret = mqtt_subscribe(&client, topic.c_str(), qos);
        return ret == MQTT_OK;
    }
    
    void setMessageHandler(std::function<void(const std::string&, const std::string&)> handler) override {
        msgHandler = handler;
    }
    
    void loop() override {
        // Not needed - handled in worker thread
    }
};

extern "C" IMessageBroker* createMQTTBroker(const char* host, int port) {
    return new MQTTBroker(host, port);
}