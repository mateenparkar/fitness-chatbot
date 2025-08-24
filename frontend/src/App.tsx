import { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [userInput, setUserInput] = useState('');
  const [chat, setChat] = useState<{ user: string; bot: string }[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chat, isTyping]);

  const handleSend = async () => {
    if (!userInput.trim()) return;

    const userMessage = userInput;
    setUserInput('');
    setChat(prev => [...prev, { user: userMessage, bot: '' }]);
    setIsTyping(true);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage }),
      });
      const data = await response.json();
      const botMessage = data.response;

      setChat(prev => {
        const newChat = [...prev];
        newChat[newChat.length - 1].bot = botMessage;
        return newChat;
      });
    } catch (error) {
      setChat(prev => {
        const newChat = [...prev];
        newChat[newChat.length - 1].bot = 'Error connecting to bot.';
        return newChat;
      });
      console.error('Error:', error);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="app-container">
      <h1>Fitness Chatbot</h1>

      <div className="chat-window">
        {chat.map((entry, index) => (
          <div key={index}>
            <div className="message user-message">{entry.user}</div>
            {entry.bot && <div className="message bot-message">{entry.bot}</div>}
          </div>
        ))}

        {isTyping && <div className="message bot-message typing">Bot is typing<span className="dot">.</span><span className="dot">.</span><span className="dot">.</span></div>}

        <div ref={chatEndRef}></div>
      </div>

      <div className="input-section">
        <input
          value={userInput}
          onChange={e => setUserInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSend()}
          placeholder="Ask a fitness question..."
        />
        <button onClick={handleSend}>Send</button>
      </div>
    </div>
  );
}

export default App;
