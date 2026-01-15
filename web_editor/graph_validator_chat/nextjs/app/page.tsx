'use client';

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import styles from './page.module.css';
import Widget from '../components/Widget';

interface Message {
  role: 'user' | 'bot';
  content: string;
  widget?: { type: string; data: any };
}

interface Triple {
  index: number;
  head: { name: string; label?: string };
  relation: string;
  tail: { name: string; label?: string };
}

interface WidgetData {
  triples?: Triple[];
  question?: string;
  triple?: { head: string; relation: string; tail: string };
  entity_name?: string;
  stats?: any;
  patent?: any;
  issues?: any[];
  suggestions?: any[];
}

const STORAGE_KEY_MESSAGES = 'chat_messages';
const STORAGE_KEY_TRIPLES = 'chat_triples';
const STORAGE_KEY_CHANGES = 'chat_changes';
const STORAGE_KEY_STATS = 'chat_stats';

export default function ChatPage() {
  const router = useRouter();
  const [mounted, setMounted] = useState(false);
  
  const [messages, setMessages] = useState<Message[]>([
    { role: 'bot', content: 'Welcome! I am setting up the environment. You can explore the sidebar while I analyze the graph.' }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [inputDisabled, setInputDisabled] = useState(true);
  const [status, setStatus] = useState('Connecting to backend...');
  const [isBackendReady, setIsBackendReady] = useState(false);
  const [allTriples, setAllTriples] = useState<Triple[]>([]);
  const [filteredTriples, setFilteredTriples] = useState<Triple[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [changes, setChanges] = useState<string[]>([]);
  const [stats, setStats] = useState<any>({});
  const chatStartedRef = useRef(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setMounted(true);
    
    // Load stored state after mount to avoid hydration errors
    try {
      const storedMessages = localStorage.getItem(STORAGE_KEY_MESSAGES);
      if (storedMessages) {
        const parsed = JSON.parse(storedMessages);
        if (parsed && parsed.length > 0) {
          setMessages(parsed);
        }
      }
      
      const storedTriples = localStorage.getItem(STORAGE_KEY_TRIPLES);
      if (storedTriples) {
        const parsed = JSON.parse(storedTriples);
        if (parsed && parsed.length > 0) {
          setAllTriples(parsed);
        }
      }
      
      const storedChanges = localStorage.getItem(STORAGE_KEY_CHANGES);
      if (storedChanges) {
        const parsed = JSON.parse(storedChanges);
        if (parsed && parsed.length > 0) {
          setChanges(parsed);
        }
      }
      
      const storedStats = localStorage.getItem(STORAGE_KEY_STATS);
      if (storedStats) {
        const parsed = JSON.parse(storedStats);
        if (parsed && Object.keys(parsed).length > 0) {
          setStats(parsed);
        }
      }
    } catch (e) {
      console.error('Failed to load stored state:', e);
    }
  }, []);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem(STORAGE_KEY_MESSAGES, JSON.stringify(messages));
    }
  }, [messages]);

  useEffect(() => {
    if (typeof window !== 'undefined' && allTriples.length > 0) {
      localStorage.setItem(STORAGE_KEY_TRIPLES, JSON.stringify(allTriples));
    }
  }, [allTriples]);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem(STORAGE_KEY_CHANGES, JSON.stringify(changes));
    }
  }, [changes]);

  useEffect(() => {
    if (typeof window !== 'undefined' && Object.keys(stats).length > 0) {
      localStorage.setItem(STORAGE_KEY_STATS, JSON.stringify(stats));
    }
  }, [stats]);

  useEffect(() => {
    // Run initial checks immediately
    checkStatus();
    updateGraphState();
    loadTriples();

    const interval = setInterval(() => {
      checkStatus();
      updateGraphState();
      loadTriples();
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    filterTriples(searchTerm);
  }, [searchTerm, allTriples]);

  const checkStatus = async () => {
    try {
      const response = await fetch('/api/status');
      const data = await response.json();
      if (data.initialized) {
        setIsBackendReady(true);
        setInputDisabled(false);
        const unanswered = data.num_unanswered !== undefined ? data.num_unanswered : data.num_questions;
        setStatus(`Ready | ${unanswered} question${unanswered !== 1 ? 's' : ''} remaining | ${data.num_triples} triples`);
        
        // Don't auto-start chat if we have stored messages (user was navigating)
        const hasStoredMessages = messages.length > 1 || (messages.length === 1 && messages[0].content !== 'Welcome! I am setting up the environment. You can explore the sidebar while I analyze the graph.');
        if (!chatStartedRef.current && !hasStoredMessages) {
          chatStartedRef.current = true;
          startChat();
        }
      } else {
        setIsBackendReady(false);
        setInputDisabled(true);
        setStatus('Initializing Backend...');
      }
    } catch (error) {
      console.error('Error checking status:', error);
      setIsBackendReady(false);
      setInputDisabled(true);
      setStatus('Waiting for server...');
    }
  };

  const startChat = async () => {
    try {
      const response = await fetch('/api/questions/first');
      const data = await response.json();
      if (data.error) {
        addMessage('bot', `Error: ${data.error}`);
        setInputDisabled(false);
      } else if (data.all_completed) {
        addMessage('bot', '✅ All questions have been answered! Graph validation is complete.');
        setInputDisabled(false);
      } else if (data.question) {
        addMessage('bot', data.question.text);
        setInputDisabled(false);
      } else {
        setInputDisabled(false);
      }
    } catch (error) {
      console.error('Error starting chat:', error);
      addMessage('bot', 'Error loading questions. Please refresh the page.');
      setInputDisabled(false);
    }
  };

  const loadTriples = async () => {
    try {
      const response = await fetch('/api/triples');
      const data = await response.json();
      if (!data.error) {
        setAllTriples(data.triples || []);
      }
    } catch (error) {
      console.error('Error loading triples:', error);
    }
  };

  const filterTriples = (term: string) => {
    const filtered = term
      ? allTriples.filter(t =>
          t.head.name.toLowerCase().includes(term.toLowerCase()) ||
          t.tail.name.toLowerCase().includes(term.toLowerCase()) ||
          t.relation.toLowerCase().includes(term.toLowerCase()) ||
          (t.head.label && t.head.label.toLowerCase().includes(term.toLowerCase())) ||
          (t.tail.label && t.tail.label.toLowerCase().includes(term.toLowerCase()))
        )
      : allTriples;
    setFilteredTriples(filtered);
  };

  const updateGraphState = async () => {
    try {
      const response = await fetch('/api/state');
      const data = await response.json();
      if (!data.error) {
        setStats({
          nodes: data.graph?.num_nodes || 0,
          edges: data.graph?.num_edges || 0,
          triples: data.num_triples || 0,
          entities: data.num_entities || 0,
        });
      }
    } catch (error) {
      console.error('Error updating state:', error);
    }
  };

  const addMessage = (role: 'user' | 'bot', content: string) => {
    setMessages(prev => [...prev, { role, content }]);
  };

  const sendAnswer = async (message?: string) => {
    const answer = message || inputValue.trim();
    if (!answer) return;

    setInputDisabled(true);
    addMessage('user', answer);
    if (!message) setInputValue('');

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: answer }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server error (${response.status}): ${errorText.substring(0, 100)}`);
      }

      const data = await response.json();
      if (data.error) {
        addMessage('bot', `Error: ${data.error}`);
        setInputDisabled(false);
      } else if (data.text && data.text.startsWith('Error:')) {
        // Handle errors returned in the text field (from validator.chat())
        console.error('Error in validator response:', data.text);
        addMessage('bot', data.text);
        setInputDisabled(false);
      } else {
        addMessage('bot', data.text || 'No response generated.');
        setChanges(data.changes_summary || []);
        setStats(prev => ({ ...prev, ...(data.stats || {}) }));

        if (data.show_widget) {
          displayWidget(data.widget_type, data.widget_data || {});
        }

        if (data.validation_complete) {
          addMessage('bot', '🎉 Graph validation complete!');
        } else {
          setInputDisabled(false);
        }

        updateGraphState();
        checkStatus();
      }
    } catch (error) {
      console.error('Error sending answer:', error);
      // Log more details about the error
      if (error instanceof Error) {
        console.error('Error message:', error.message);
        console.error('Error stack:', error.stack);
      }
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      addMessage('bot', `Error processing answer: ${errorMessage}. Please try again.`);
      setInputDisabled(false);
    }
  };

  const displayWidget = (widgetType: string, widgetData: WidgetData) => {
    // Add widget to messages
    setMessages(prev => [...prev, { 
      role: 'bot', 
      content: '',
      widget: { type: widgetType, data: widgetData }
    }]);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !inputDisabled) {
      sendAnswer();
    }
  };

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <button
          className={styles.navButton}
          onClick={() => router.push('/widget-showcase')}
          title="Go to Widget Showcase"
        >
          ←
        </button>
        <div className={styles.headerContent}>
          <h1>Graph Validator Chat</h1>
          <div className={styles.status}>{status}</div>
        </div>
        <button
          className={styles.navButton}
          onClick={() => router.push('/graph')}
          title="Go to Graph"
        >
          →
        </button>
      </header>
      <div className={styles.chatContainer}>
        <div className={styles.messages}>
          {messages.map((msg, idx) => (
            <div key={idx}>
              {msg.content && (
                <div className={`${styles.message} ${styles[msg.role]}`}>
                  <div className={styles.messageContent}>
                    <strong>{msg.role === 'bot' ? 'Bot' : 'You'}:</strong> {msg.content}
                  </div>
                </div>
              )}
              {msg.widget && (
                <div className={styles.widgetContainer}>
                  <Widget
                    type={msg.widget.type}
                    data={msg.widget.data}
                    onAnswer={(answer) => {
                      sendAnswer(answer);
                    }}
                  />
                </div>
              )}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
        <div className={styles.inputArea}>
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message here..."
            disabled={inputDisabled}
            className={styles.input}
          />
          <button
            onClick={() => sendAnswer()}
            disabled={inputDisabled}
            className={styles.sendButton}
          >
            Send
          </button>
        </div>
      </div>
      <div className={styles.sidebar}>
        <h3>Triples</h3>
        <div className={styles.triplesContainer}>
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search triples..."
            className={styles.tripleSearch}
          />
          <div className={styles.triplesList}>
            {filteredTriples.length === 0 ? (
              <p style={{ color: '#999', fontStyle: 'italic' }}>No triples found</p>
            ) : (
              <div className={styles.triplesScroll}>
                {filteredTriples.map((triple, idx) => (
                  <div key={idx} className={styles.tripleWidget}>
                    <div className={styles.tripleIndex}>#{triple.index}</div>
                    <div className={styles.tripleContent}>
                      <div className={styles.tripleHead}>
                        <span className={styles.entityName}>{triple.head.name}</span>
                        {triple.head.label && (
                          <span className={styles.entityLabel}>{triple.head.label}</span>
                        )}
                      </div>
                      <div className={styles.tripleRelation}>{triple.relation}</div>
                      <div className={styles.tripleTail}>
                        <span className={styles.entityName}>{triple.tail.name}</span>
                        {triple.tail.label && (
                          <span className={styles.entityLabel}>{triple.tail.label}</span>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
        <h3>Recent Changes</h3>
        <div className={styles.stateDisplay}>
          {changes.length === 0 ? (
            <p style={{ color: '#999', fontStyle: 'italic' }}>No changes yet</p>
          ) : (
            <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '13px' }}>
              {changes.slice(0, 5).map((change, idx) => (
                <li key={idx} style={{ marginBottom: '6px' }}>{change}</li>
              ))}
              {changes.length > 5 && (
                <li style={{ color: '#999', fontStyle: 'italic' }}>
                  ... and {changes.length - 5} more
                </li>
              )}
            </ul>
          )}
        </div>
        <h3>Graph Statistics</h3>
        <div className={styles.stateDisplay}>
          <div style={{ fontSize: '13px', lineHeight: '1.8' }}>
            <p><strong>Nodes:</strong> {stats.nodes || 0}</p>
            <p><strong>Edges:</strong> {stats.edges || 0}</p>
            <p><strong>Triples:</strong> {stats.triples || 0}</p>
            <p><strong>Entities:</strong> {stats.entities || 0}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

