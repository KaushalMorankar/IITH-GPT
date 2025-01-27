'use client';
import { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
export default function Chatbot() {
  const [messages, setMessages] = useState([
    { text: "Hello! How can I help you today?", sender: "bot" },
  ]);
  const [userInput, setUserInput] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    if (userInput.trim() === "") return;

    const newMessages = [...messages, { text: userInput, sender: "user" }];
    setMessages(newMessages);
    setUserInput("");

    setLoading(true);
    try {
      const botResponse = await getBotResponse(userInput);
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: botResponse, sender: "bot" },
      ]);
      console.log("Bot response:", botResponse);
    } catch (error) {
      console.error("Error fetching bot response:", error);
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: "Oops! Something went wrong. Please try again later.", sender: "bot" },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const getBotResponse = async (input) => {
    const response = await fetch("http://localhost:5000/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query: input.trim() }), // Trim input to remove extra spaces
    });

    if (!response.ok) {
      throw new Error("Failed to get response from server");
    }

    const data = await response.json();
    return data.response;
  };

  useEffect(() => {
    const chatBox = document.querySelector("[style*='overflowY: scroll']");
    if (chatBox) chatBox.scrollTop = chatBox.scrollHeight;
  }, [messages]);

  return (
    <div style={styles.container}>
      <div style={styles.chatBox}>
      {messages.map((message, index) => (
  <div
    key={index}
    style={{
      ...styles.message,
      alignSelf: message.sender === "user" ? "flex-end" : "flex-start",
      backgroundColor: message.sender === "user" ? "#f0f0f0" : "#d1f7d6",
      borderRadius: message.sender === "user" 
        ? "15px 15px 0px 15px" 
        : "15px 15px 15px 0px",
    }}
  >
    <strong>{message.sender === "user" ? "You: " : "IITH_GPT: "}</strong>
    <ReactMarkdown>{message.text}</ReactMarkdown>
  </div>
))}
      
        {loading && (
          <div style={{ ...styles.message, alignSelf: "flex-start" }}>
            Typing...
          </div>
        )}
      </div>
      <div style={styles.inputContainer}>
        <input
          type="text"
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          placeholder="Type your message..."
          style={styles.input}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          disabled={loading}
        />
        <button onClick={handleSend} style={styles.button} disabled={loading}>
          {loading ? "Sending..." : "Send"}
        </button>
      </div>
    </div>
  );
}

// Updated styles to match provided UI

const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    height: "100vh",
    width: "100vw",
    fontFamily: "Arial, sans-serif",
    backgroundImage: "url('data:image/jpeg;base64,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')",
    backgroundSize: "cover",
    backgroundPosition: "center",
    backgroundRepeat: "no-repeat",
    padding: "0",
    margin: "0",
  },
  chatBox: {
    width: "80vw",
    height: "75vh",
    border: "1px solid #ccc",
    borderRadius: "8px",
    padding: "20px",
    overflowY: "scroll",
    display: "flex",
    flexDirection: "column",
    gap: "20px",
    backgroundColor: "rgba(255, 255, 255, 0.1)",  // Semi-transparent background
    backdropFilter: "blur(8px)",
    boxShadow: "0 4px 8px rgba(0, 0, 0, 0.2)",
  },
  message: {
    maxWidth: "70%",
    padding: "15px",
    borderRadius: "10px",
    fontSize: "16px",
    wordWrap: "break-word",
    color: "#000",
    boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)",
  },
  inputContainer: {
    display: "flex",
    width: "80vw",
    marginTop: "10px",
  },
  input: {
    flex: 1,
    padding: "15px",
    border: "1px solid #ccc",
    borderRadius: "8px 0 0 8px",
    outline: "none",
    fontSize: "16px",
  },
  button: {
    padding: "15px 20px",
    backgroundColor: "#0070f3",
    color: "white",
    border: "none",
    borderRadius: "0 8px 8px 0",
    cursor: "pointer",
    fontSize: "16px",
  },
};
