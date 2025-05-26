import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { ChevronLeft, Send, Lightbulb, Mic, Upload } from 'lucide-react';
import axios from 'axios';
import './ChatPage.css';

const suggestions = [
  "Check my loan eligibility",
  "Guide me through loan application",
  "Show me financial tips",
  "Explain loan terms in my language"
];

function ChatPage() {
  const navigate = useNavigate();
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([
    {
      type: 'assistant',
      content: "Hello! I'm your Loan Advisor AI. How can I assist you with your financial journey today?"
    }
  ]);
  const [userDetails, setUserDetails] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  useEffect(() => {
    const fetchUserDetails = async () => {
      try {
        const token = localStorage.getItem('token');
        if (!token) {
          console.error('No token found in localStorage');
          navigate('/login');
          return;
        }

        const response = await axios.get('http://localhost:5001/api/user-details', {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        });
        setUserDetails(response.data.userDetails);
      } catch (error) {
        console.error('Error fetching user details:', error.response ? error.response.data : error.message);
        if (error.response && error.response.status === 401) {
          localStorage.removeItem('token');
          navigate('/login');
        }
      }
    };
    fetchUserDetails();
  }, [navigate]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        setAudioBlob(audioBlob);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      console.log('Recording started');
    } catch (error) {
      console.error('Error starting recording:', error);
      setMessages((prev) => [
        ...prev,
        { type: 'assistant', content: 'Failed to start recording. Please allow microphone access.' }
      ]);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      console.log('Recording stopped');
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file && (file.type === 'audio/wav' || file.type === 'audio/mpeg')) {
      setAudioBlob(file);
      console.log('Audio file uploaded:', file.name);
    } else {
      setMessages((prev) => [
        ...prev,
        { type: 'assistant', content: 'Please upload a valid WAV or MP3 file.' }
      ]);
    }
  };

  const handleSendText = async () => {
    if (!input.trim()) return;

    setMessages((prev) => [
      ...prev,
      { type: 'user', content: input },
      { type: 'assistant', content: 'Processing your request...' }
    ]);
    setInput('');

    try {
      const token = localStorage.getItem('token');
      if (!token) {
        throw new Error('No token found. Please log in again.');
      }

      const response = await axios.post(
        'http://localhost:5001/api/chat',
        { message: input },
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        }
      );

      setMessages((prev) => [
        ...prev.slice(0, -1),
        { type: 'assistant', content: response.data.response }
      ]);
    } catch (error) {
      console.error('Error sending text message:', error.response ? error.response.data : error.message);
      let errorMessage = 'Error connecting to the chat service. Please try again.';

      if (error.response) {
        if (error.response.status === 401) {
          errorMessage = error.response.data.error || 'Session expired. Please log in again.';
          localStorage.removeItem('token');
          navigate('/login');
        } else if (error.response.status === 400) {
          errorMessage = error.response.data.error || 'Invalid request. Please check your input.';
        } else {
          errorMessage = error.response.data.error || errorMessage;
        }
      }

      setMessages((prev) => [
        ...prev.slice(0, -1),
        { type: 'assistant', content: errorMessage }
      ]);
    }
  };

  const handleSendAudio = async () => {
    if (!audioBlob) {
      setMessages((prev) => [
        ...prev,
        { type: 'assistant', content: 'Please record or upload an audio file first.' }
      ]);
      return;
    }

    setMessages((prev) => [
      ...prev,
      { type: 'user', content: '🎙️ Audio message sent' },
      { type: 'assistant', content: 'Processing your audio request...' }
    ]);

    try {
      const token = localStorage.getItem('token');
      if (!token) {
        throw new Error('No token found. Please log in again.');
      }

      const formData = new FormData();
      formData.append('audio', audioBlob, 'audio.wav');

      const response = await axios.post(
        'http://localhost:5001/api/voice-chat',
        formData,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'multipart/form-data'
          }
        }
      );

      setMessages((prev) => [
        ...prev.slice(0, -1),
        { type: 'assistant', content: response.data.response_html },
        {
          type: 'assistant',
          content: (
            <audio controls autoPlay>
              <source src={`data:audio/mpeg;base64,${response.data.response_audio}`} type="audio/mpeg" />
              Your browser does not support the audio element.
            </audio>
          )
        }
      ]);
      setAudioBlob(null);
    } catch (error) {
      console.error('Error sending audio message:', error.response ? error.response.data : error.message);
      let errorMessage = 'Error processing audio. Please try again.';

      if (error.response) {
        if (error.response.status === 401) {
          errorMessage = error.response.data.error || 'Session expired. Please log in again.';
          localStorage.removeItem('token');
          navigate('/login');
        } else if (error.response.status === 400) {
          errorMessage = error.response.data.error || 'Invalid audio file. Please upload WAV or MP3.';
        } else {
          errorMessage = error.response.data.error || errorMessage;
        }
      }

      setMessages((prev) => [
        ...prev.slice(0, -1),
        { type: 'assistant', content: errorMessage }
      ]);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendText();
    }
  };

  return (
    <div className="min-h-screen bg-white flex flex-col">
      <div className="bg-white border-b border-gray-200 shadow-sm sticky top-0 z-20">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <button
            onClick={() => navigate('/')}
            className="btn btn-ghost text-gray-600 gap-2 hover:bg-gray-100 transition"
          >
            <ChevronLeft size={24} className="text-gray-600" />
            <span className="text-lg font-semibold">Back to Home</span>
          </button>
        </div>
      </div>

      <div className="flex-1 max-w-7xl w-full mx-auto px-6 py-8 flex flex-col">
        <div className="flex-1 space-y-6 mb-8 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-400 scrollbar-track-gray-100">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex animate-fade-in-up ${
                message.type === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              <div
                className={`max-w-[70%] rounded-xl p-4 ${
                  message.type === 'user'
                    ? 'bg-indigo-500 text-white shadow-md shadow-gray-300'
                    : 'bg-gray-50 text-gray-900 shadow-md shadow-gray-200'
                }`}
              >
                {message.type === 'user' ? (
                  <p className="text-white">{message.content}</p>
                ) : typeof message.content === 'string' ? (
                  <div dangerouslySetInnerHTML={{ __html: message.content }} />
                ) : (
                  message.content
                )}
              </div>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {suggestions.map((suggestion, index) => (
            <button
              key={index}
              className="btn btn-outline text-gray-700 border-gray-300 hover:bg-gray-100 hover:border-gray-400 gap-2 transition-all shadow-sm shadow-gray-200"
              onClick={() => setInput(suggestion)}
            >
              <Lightbulb size={16} className="text-yellow-500" />
              <span className="text-sm">{suggestion}</span>
            </button>
          ))}
        </div>

        <div className="bg-white rounded-xl shadow-lg shadow-gray-300 p-4 border border-gray-200">
          <div className="flex flex-col gap-4">
            <div className="flex gap-4 items-center">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask me anything about loans or finances..."
                className="flex-1 textarea bg-gray-50 text-gray-900 placeholder-gray-500 border-gray-300 focus:border-indigo-500 focus:ring-0 resize-none scrollbar-thin scrollbar-thumb-gray-400 scrollbar-track-gray-100"
                rows={2}
              />
              <button
                className="btn btn-circle bg-indigo-500 hover:bg-indigo-600 text-white border-none"
                onClick={handleSendText}
                disabled={!input.trim()}
              >
                <Send size={24} />
              </button>
            </div>
            <div className="flex gap-4 items-center">
              <button
                className={`btn btn-circle ${isRecording ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600'} text-white border-none`}
                onClick={isRecording ? stopRecording : startRecording}
              >
                <Mic size={24} />
              </button>
              <label className="btn btn-circle bg-blue-500 hover:bg-blue-600 text-white border-none cursor-pointer">
                <Upload size={24} />
                <input
                  type="file"
                  accept="audio/wav,audio/mpeg"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </label>
              {audioBlob && (
                <button
                  className="btn btn-circle bg-indigo-500 hover:bg-indigo-600 text-white border-none"
                  onClick={handleSendAudio}
                >
                  <Send size={24} />
                </button>
              )}
            </div>
            {audioBlob && (
              <div className="text-sm text-gray-600 flex items-center gap-2">
                Audio ready: {audioBlob.name || 'Recorded audio'}
                <audio controls src={URL.createObjectURL(audioBlob)} className="h-8" />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default ChatPage;