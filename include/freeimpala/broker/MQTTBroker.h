#pragma once

#include <string>
#include <functional>
#include <thread>
#include <atomic>
#include <vector>
#include <cstring> // For std::memcpy (though not directly used, good to have for string manipulation)
#include <cerrno>  // For errno
#include <chrono>  // For std::chrono::milliseconds

/* 1. pull in MQTT-C with C linkage */
extern "C" {
#include "mqtt_c/mqtt.h" // Assuming this path is correct relative to where this header will be used
}

#include <fcntl.h>      // For fcntl, O_NONBLOCK
#include <netdb.h>      // For getaddrinfo, addrinfo
#include <sys/socket.h> // For socket, connect, AF_UNSPEC, SOCK_STREAM
#include <unistd.h>     // For close

// Assuming IMessageBroker.h exists and defines the interface
#include "IMessageBroker.h"

/**
 * @class MQTTBroker
 * @brief Implements IMessageBroker using the MQTT-C library for MQTT communication.
 *
 * This class provides a header-only implementation for an MQTT client,
 * allowing it to connect to an MQTT broker, publish messages, subscribe to topics,
 * and handle incoming messages. It wraps the C-based MQTT-C library with a C++ interface.
 */
class MQTTBroker : public IMessageBroker {
public:
    /**
     * @brief Constructs an MQTTBroker instance and attempts to connect to the specified broker.
     * @param host The hostname or IP address of the MQTT broker. Defaults to "localhost".
     * @param port The port number of the MQTT broker. Defaults to "1883".
     * @param cid The client ID to use for the MQTT connection. Defaults to "mqtt_cxx_client".
     * @param keep_alive The keep-alive interval in seconds. Defaults to 400.
     * @throws std::runtime_error if socket connection or MQTT connection fails.
     */
    explicit MQTTBroker(std::string  host   = "localhost",
                        std::string  port   = "1883",
                        std::string  cid    = "mqtt_cxx_client",
                        uint16_t     keep_alive = 400)
        : host_(std::move(host))
        , port_(std::move(port))
        , client_id_(std::move(cid))
        , keep_alive_(keep_alive)
        , sendbuf_(2048) // Initialize send buffer with a default size
        , recvbuf_(2048) // Initialize receive buffer with a default size
    {
        // Attempt to open a non-blocking socket and connect to the broker
        sock_ = open_nb_socket(host_.c_str(), port_.c_str());
        if (sock_ < 0) {
            throw std::runtime_error("MQTTBroker: socket/connect failed");
        }

        // Initialize the MQTT client with the socket and buffers
        mqtt_init(&client_,
                  sock_,
                  sendbuf_.data(), sendbuf_.size(),
                  recvbuf_.data(), recvbuf_.size(),
                  &MQTTBroker::s_on_publish); // Set the static publish callback

        // Make 'this' pointer available to the static callback function
        client_.publish_response_callback_state = this;

        // Optional: Force a clean session for the MQTT connection
        uint8_t flags = MQTT_CONNECT_CLEAN_SESSION;

        // Connect to the MQTT broker
        auto rc = mqtt_connect(&client_,
                               client_id_.c_str(),    /* client_id */
                               nullptr, nullptr, 0,    /* no will */
                               nullptr, nullptr,       /* no user/pass */
                               flags,
                               keep_alive_);
        if (rc != MQTT_OK) {
            // Close the socket if connection fails
            ::close(sock_);
            sock_ = -1; // Invalidate socket descriptor
            throw std::runtime_error("MQTTBroker: mqtt_connect failed with error code " + std::to_string(rc));
        }
    }

    /**
     * @brief Destructor for MQTTBroker. Closes the socket if it's open.
     */
    ~MQTTBroker() override {
        if (sock_ >= 0) {
            ::close(sock_);
        }
    }

    /**
     * @brief Publishes a message to a given MQTT topic.
     * @param topic The topic to publish to.
     * @param message The message payload.
     * @return True if the message was successfully queued for publishing, false otherwise.
     */
    bool publish(const std::string& topic,
                 const std::string& message) override {
        int rc = mqtt_publish(&client_,
                              topic.c_str(),
                              message.data(),
                              message.size(),
                              MQTT_PUBLISH_QOS_0); // Using QoS 0 for simplicity
        return rc == MQTT_OK;
    }

    /**
     * @brief Subscribes to an MQTT topic.
     * @param topic The topic to subscribe to.
     * @return True if the subscription request was successfully sent, false otherwise.
     */
    bool subscribe(const std::string& topic) override {
        // Subscribe with maximum QoS 0 (can be adjusted if needed)
        int rc = mqtt_subscribe(&client_, topic.c_str(), 0);
        return rc == MQTT_OK;
    }

    /**
     * @brief Sets the callback function to be invoked when a message is received.
     * @param handler A std::function that takes two std::string arguments (topic, message).
     */
    void setMessageHandler(std::function<void(const std::string&,
                                              const std::string&)> handler) override {
        user_cb_ = std::move(handler);
    }

    /**
     * @brief Drives the MQTT-C event loop. This method should be called periodically
     * to process incoming and outgoing MQTT packets.
     * It performs a non-blocking synchronization and then sleeps for 100ms.
     * If `mqtt_sync` fails, it indicates a potential connection issue.
     */
    void loop() override {
        // Non-blocking sync; recommended cadence ≈10 Hz (every 100ms)
        if (mqtt_sync(&client_) != MQTT_OK) {
            // In a real application, you might implement reconnection logic here.
            // For now, we just acknowledge the failure.
            // std::cerr << "MQTTBroker: mqtt_sync failed, connection might be lost." << std::endl;
        }
        // Introduce a small delay to prevent busy-waiting and reduce CPU usage
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

private:
    /* raw helpers ---------------------------------------------------------- */
    /**
     * @brief Utility function to open a non-blocking POSIX TCP socket and connect.
     * Adapted from MQTT-C’s examples/templates/posix_sockets.h.
     * @param addr The address (hostname or IP) to connect to.
     * @param port The port number to connect to.
     * @return The socket file descriptor on success, or -1 on failure.
     */
    static int open_nb_socket(const char* addr, const char* port) {
        struct addrinfo hints {};
        hints.ai_family   = AF_UNSPEC;     /* IPv4 / IPv6 */
        hints.ai_socktype = SOCK_STREAM;   /* TCP */

        struct addrinfo* res;
        int rv = getaddrinfo(addr, port, &hints, &res);
        if (rv != 0) {
            // fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(rv));
            return -1;
        }

        int sock = -1;
        for (auto* p = res; p; p = p->ai_next) {
            sock = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
            if (sock == -1) {
                continue;
            }
            // Set socket to non-blocking mode
            int flags = fcntl(sock, F_GETFL, 0);
            fcntl(sock, F_SETFL, flags | O_NONBLOCK);

            // Attempt to connect. If it returns 0, connection is immediate.
            // If errno is EINPROGRESS, connection is in progress (non-blocking).
            if (::connect(sock, p->ai_addr, p->ai_addrlen) == 0 ||
                errno == EINPROGRESS) {
                break; // Connected (or in progress)
            }
            ::close(sock); // Close socket if connect failed immediately
            sock = -1;     // Reset sock to indicate failure for this address
        }
        freeaddrinfo(res); // Free the address info structure
        return sock;
    }

    /**
     * @brief Static callback function for MQTT-C to handle incoming publish messages.
     * This function retrieves the `MQTTBroker` instance from the `state` pointer
     * and then calls the user-defined message handler.
     * @param state A pointer to the `MQTTBroker` instance (set via `client_.publish_response_callback_state`).
     * @param pub A pointer to the `mqtt_response_publish` structure containing message details.
     */
    static void s_on_publish(void** state,
                             struct mqtt_response_publish* pub) {
        // Cast the void** state back to MQTTBroker*
        auto* self = static_cast<MQTTBroker*>(*state);
        // Ensure self, user_cb_, and pub are valid before proceeding
        if (!self || !self->user_cb_ || !pub) {
            return;
        }

        // Extract topic and payload from the MQTT-C structure
        // Cast to const char* to avoid casting away const qualifier
        std::string topic(reinterpret_cast<const char*>(pub->topic_name),
                          pub->topic_name_size);
        std::string payload(reinterpret_cast<const char*>(pub->application_message),
                            pub->application_message_size);

        // Invoke the user's message handler
        self->user_cb_(topic, payload);
    }

    /* configuration -------------------------------------------------------- */
    const std::string host_;        ///< MQTT broker hostname or IP address
    const std::string port_;        ///< MQTT broker port
    const std::string client_id_;   ///< MQTT client ID
    uint16_t          keep_alive_;  ///< MQTT keep-alive interval

    /* MQTT-C objects ------------------------------------------------------- */
    int                 sock_      = -1; ///< Socket file descriptor
    mqtt_client         client_{};      ///< MQTT-C client structure
    std::vector<uint8_t> sendbuf_;     ///< Buffer for outgoing MQTT packets
    std::vector<uint8_t> recvbuf_;     ///< Buffer for incoming MQTT packets

    /* user callback -------------------------------------------------------- */
    std::function<void(const std::string&,
                       const std::string&)> user_cb_; ///< User-defined message handler
};
