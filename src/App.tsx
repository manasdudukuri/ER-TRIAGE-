/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { GoogleGenAI, Type } from "@google/genai";
import { 
  Activity, 
  AlertCircle, 
  Clock, 
  History, 
  Send, 
  Stethoscope, 
  User, 
  ShieldAlert,
  CheckCircle2,
  Loader2,
  Trash2,
  Plus,
  Mic,
  MicOff,
  Heart,
  Zap,
  Building2,
  ArrowRightLeft,
  Timer
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

// Utility for tailwind classes
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Types
interface TriageResult {
  patient_id: string;
  age: number | null;
  gender: string | null;
  primary_symptoms: string[];
  esi_level: number;
  effective_esi: number; // For aging algorithm
  required_resources: string[];
  justification: string;
  timestamp: string;
  vitals?: {
    hr: number;
    spo2: number;
    status: 'stable' | 'critical';
  };
  wait_minutes: number;
}

const SYSTEM_INSTRUCTION = `You are an expert Emergency Room Triage AI operating in a high-stress hospital environment. 
Your job is to read messy, unstructured notes (or listen to audio) from intake nurses and instantly classify the patient's severity using the standard Emergency Severity Index (ESI), where 1 is Critical/Life-Saving and 5 is Non-Urgent.

RULES:
1. You MUST output ONLY valid JSON.
2. Extract the patient's age, gender, and primary symptoms. If age or gender is unknown, return null.
3. Determine the ESI Level (1 to 5).
4. List the immediate hospital resources required. Be specific and granular.
5. Provide a brief, clinical 1-sentence justification for your assigned ESI level.`;

const RESPONSE_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    patient_id: { type: Type.STRING },
    age: { type: Type.INTEGER, nullable: true },
    gender: { type: Type.STRING, nullable: true },
    primary_symptoms: { type: Type.ARRAY, items: { type: Type.STRING } },
    esi_level: { type: Type.INTEGER },
    required_resources: { type: Type.ARRAY, items: { type: Type.STRING } },
    justification: { type: Type.STRING }
  },
  required: ["patient_id", "esi_level", "primary_symptoms", "required_resources", "justification"]
};

const MAX_CAPACITY = 10; // Demo limit

export default function App() {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [queue, setQueue] = useState<TriageResult[]>([]);
  const [currentResult, setCurrentResult] = useState<TriageResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [diversionAlert, setDiversionAlert] = useState<string | null>(null);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  // --- 1. ANTI-STARVATION AGING ALGORITHM ---
  // Every 10 seconds (simulated 30 mins), increase priority of waiting patients
  useEffect(() => {
    const interval = setInterval(() => {
      setQueue(prevQueue => {
        const updated = prevQueue.map(p => {
          const newWait = p.wait_minutes + 1;
          // Aging: If waiting > 2 cycles, boost priority (lower effective ESI)
          let effectiveEsi = p.esi_level;
          if (newWait >= 3 && effectiveEsi > 1) effectiveEsi -= 1;
          if (newWait >= 6 && effectiveEsi > 1) effectiveEsi -= 1;
          
          return { ...p, wait_minutes: newWait, effective_esi: effectiveEsi };
        });
        // Sort by effective ESI, then by wait time
        return [...updated].sort((a, b) => a.effective_esi - b.effective_esi || b.wait_minutes - a.wait_minutes);
      });
    }, 10000);
    return () => clearInterval(interval);
  }, []);

  // --- 3. IOT CRASH INTERRUPTS SIMULATION ---
  useEffect(() => {
    const interval = setInterval(() => {
      setQueue(prevQueue => {
        return prevQueue.map(p => {
          // 5% chance of a "crash" for non-critical patients
          if (p.esi_level > 1 && Math.random() < 0.05 && p.vitals?.status !== 'critical') {
            return {
              ...p,
              vitals: { hr: 145, spo2: 82, status: 'critical' },
              effective_esi: 1, // Instant priority override
              justification: "CRITICAL INTERRUPT: IoT Vitals detected acute decompensation (SpO2 < 85%)."
            };
          }
          // Normal vitals jitter
          return {
            ...p,
            vitals: p.vitals ? {
              ...p.vitals,
              hr: p.vitals.status === 'critical' ? 140 + Math.random() * 10 : 70 + Math.random() * 20,
              spo2: p.vitals.status === 'critical' ? 80 + Math.random() * 5 : 95 + Math.random() * 4,
            } : { hr: 75, spo2: 98, status: 'stable' }
          };
        });
      });
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  // --- 2. HANDS-FREE AUDIO INTAKE ---
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        await handleTriage(audioBlob);
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      setError("Microphone access denied.");
    }
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    setIsRecording(false);
  };

  const handleTriage = async (audioBlob?: Blob) => {
    // --- 4. CAPACITY OVERFLOW ---
    if (queue.length >= MAX_CAPACITY) {
      setDiversionAlert("HOSPITAL AT CAPACITY. DIVERSION PROTOCOL ACTIVE: Routing to St. Jude Medical Center.");
      return;
    }

    setLoading(true);
    setError(null);
    setDiversionAlert(null);
    
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || '' });
      let contents: any;

      if (audioBlob) {
        const reader = new FileReader();
        const base64Promise = new Promise<string>((resolve) => {
          reader.onloadend = () => resolve((reader.result as string).split(',')[1]);
          reader.readAsDataURL(audioBlob);
        });
        const base64Data = await base64Promise;
        contents = {
          parts: [
            { inlineData: { data: base64Data, mimeType: "audio/wav" } },
            { text: "Triage this audio intake." }
          ]
        };
      } else {
        if (!input.trim()) return;
        contents = input;
      }

      const response = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: contents,
        config: {
          systemInstruction: SYSTEM_INSTRUCTION,
          responseMimeType: "application/json",
          responseSchema: RESPONSE_SCHEMA,
          temperature: 0.1,
        },
      });

      const resultData = JSON.parse(response.text || '{}');
      const fullResult: TriageResult = {
        ...resultData,
        timestamp: new Date().toISOString(),
        effective_esi: resultData.esi_level,
        wait_minutes: 0,
        vitals: { hr: 75, spo2: 98, status: 'stable' }
      };

      setCurrentResult(fullResult);
      setQueue(prev => [...prev, fullResult].sort((a, b) => a.effective_esi - b.effective_esi));
      setInput('');
      
      scrollRef.current?.scrollIntoView({ behavior: 'smooth' });

    } catch (err) {
      console.error("Triage Error:", err);
      setError("Failed to process triage. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const getESIColor = (level: number) => {
    switch (level) {
      case 1: return "bg-red-600 text-white border-red-700 shadow-red-200";
      case 2: return "bg-orange-500 text-white border-orange-600 shadow-orange-200";
      case 3: return "bg-yellow-400 text-black border-yellow-500 shadow-yellow-100";
      case 4: return "bg-green-500 text-white border-green-600 shadow-green-200";
      case 5: return "bg-blue-500 text-white border-blue-600 shadow-blue-200";
      default: return "bg-gray-500 text-white border-gray-600";
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 font-sans text-slate-100 selection:bg-red-500/30">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-slate-900/80 backdrop-blur-md border-b border-slate-800 px-6 py-4 flex items-center justify-between shadow-2xl">
        <div className="flex items-center gap-3">
          <div className="bg-red-600 p-2 rounded-lg shadow-lg shadow-red-900/20 animate-pulse">
            <Activity className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-black tracking-tighter text-white">ER TRIAGE <span className="text-red-500">PRO</span></h1>
            <p className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em]">Multimodal • IoT • Anti-Starvation</p>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-3 px-4 py-2 bg-slate-800 rounded-full border border-slate-700">
            <Building2 className={cn("w-4 h-4", queue.length >= MAX_CAPACITY ? "text-red-500" : "text-green-500")} />
            <div className="flex flex-col">
              <span className="text-[10px] font-bold text-slate-500 leading-none">CAPACITY</span>
              <span className="text-xs font-black">{queue.length} / {MAX_CAPACITY}</span>
            </div>
          </div>
          <button 
            onClick={() => setQueue([])}
            className="p-2 text-slate-500 hover:text-red-500 transition-colors"
          >
            <Trash2 className="w-5 h-5" />
          </button>
        </div>
      </header>

      {/* Diversion Alert */}
      <AnimatePresence>
        {diversionAlert && (
          <motion.div 
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="bg-red-600 text-white px-6 py-3 flex items-center justify-center gap-3 font-black text-sm uppercase tracking-wider"
          >
            <ArrowRightLeft className="w-5 h-5 animate-bounce" />
            {diversionAlert}
          </motion.div>
        )}
      </AnimatePresence>

      <main className="max-w-7xl mx-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Intake */}
        <div className="lg:col-span-5 space-y-6">
          <section className="bg-slate-900 rounded-3xl border border-slate-800 shadow-2xl overflow-hidden">
            <div className="p-6 border-b border-slate-800 bg-slate-900/50 flex items-center justify-between">
              <h2 className="flex items-center gap-2 text-xs font-black text-slate-400 uppercase tracking-widest">
                <Zap className="w-4 h-4 text-yellow-500" />
                Rapid Intake
              </h2>
              {isRecording && (
                <span className="flex items-center gap-2 text-red-500 text-[10px] font-black animate-pulse">
                  <div className="w-2 h-2 bg-red-500 rounded-full" />
                  RECORDING LIVE
                </span>
              )}
            </div>
            
            <div className="p-6 space-y-6">
              <div className="relative">
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Type notes or use Hands-Free Audio..."
                  className="w-full h-48 p-6 bg-slate-950 rounded-2xl border border-slate-800 focus:ring-2 focus:ring-red-500 focus:border-transparent outline-none transition-all resize-none text-slate-200 placeholder:text-slate-700 font-medium text-lg"
                />
                
                {/* Audio Button Overlay */}
                <button
                  onClick={isRecording ? stopRecording : startRecording}
                  className={cn(
                    "absolute bottom-4 right-4 p-4 rounded-2xl transition-all shadow-xl",
                    isRecording 
                      ? "bg-red-600 text-white scale-110 animate-pulse" 
                      : "bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-white"
                  )}
                >
                  {isRecording ? <MicOff className="w-6 h-6" /> : <Mic className="w-6 h-6" />}
                </button>
              </div>

              <button
                onClick={() => handleTriage()}
                disabled={loading || (!input.trim() && !isRecording)}
                className={cn(
                  "w-full flex items-center justify-center gap-3 py-5 rounded-2xl font-black text-lg transition-all shadow-2xl",
                  loading || (!input.trim() && !isRecording)
                    ? "bg-slate-800 text-slate-600 cursor-not-allowed" 
                    : "bg-red-600 text-white hover:bg-red-500 active:scale-95 shadow-red-900/20"
                )}
              >
                {loading ? <Loader2 className="w-6 h-6 animate-spin" /> : <Send className="w-6 h-6" />}
                {loading ? "PROCESSING..." : "SUBMIT CASE"}
              </button>
            </div>
          </section>

          {/* Current Result Card */}
          <AnimatePresence>
            {currentResult && (
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className="bg-slate-900 rounded-3xl border border-slate-800 p-6 shadow-2xl"
              >
                <div className="flex items-center justify-between mb-6">
                  <div className={cn("px-4 py-2 rounded-xl text-xs font-black uppercase tracking-tighter", getESIColor(currentResult.esi_level))}>
                    ESI {currentResult.esi_level}
                  </div>
                  <span className="text-[10px] font-bold text-slate-500">ID: {currentResult.patient_id}</span>
                </div>
                <p className="text-slate-300 font-medium italic mb-4 leading-relaxed">
                  "{currentResult.justification}"
                </p>
                <div className="flex flex-wrap gap-2">
                  {currentResult.required_resources.slice(0, 3).map((r, i) => (
                    <span key={i} className="text-[10px] font-bold px-2 py-1 bg-slate-800 text-slate-400 rounded-md border border-slate-700">
                      {r}
                    </span>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Right Column: Live Queue */}
        <div className="lg:col-span-7">
          <section className="bg-slate-900 rounded-3xl border border-slate-800 shadow-2xl h-full flex flex-col overflow-hidden">
            <div className="p-6 border-b border-slate-800 flex items-center justify-between bg-slate-900/50">
              <div className="flex items-center gap-3">
                <h2 className="text-xs font-black text-slate-400 uppercase tracking-widest">Live Triage Queue</h2>
                <div className="flex items-center gap-1.5 px-2 py-1 bg-slate-800 rounded-lg">
                  <Timer className="w-3 h-3 text-blue-400" />
                  <span className="text-[10px] font-bold text-slate-300">AGING ACTIVE</span>
                </div>
              </div>
              <div className="flex items-center gap-4 text-[10px] font-bold text-slate-500">
                <span className="flex items-center gap-1"><div className="w-2 h-2 bg-red-500 rounded-full" /> CRITICAL</span>
                <span className="flex items-center gap-1"><div className="w-2 h-2 bg-blue-500 rounded-full" /> STABLE</span>
              </div>
            </div>

            <div className="flex-1 overflow-y-auto p-6 space-y-4 max-h-[calc(100vh-200px)]">
              {queue.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-slate-700 gap-4">
                  <Activity className="w-16 h-16 opacity-10" />
                  <p className="font-black uppercase tracking-widest text-sm">No Patients in Queue</p>
                </div>
              ) : (
                <AnimatePresence>
                  {queue.map((patient, idx) => (
                    <motion.div
                      key={patient.timestamp}
                      layout
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className={cn(
                        "p-5 rounded-2xl border transition-all relative overflow-hidden",
                        patient.vitals?.status === 'critical' 
                          ? "bg-red-950/30 border-red-500/50 shadow-lg shadow-red-900/20" 
                          : "bg-slate-950 border-slate-800"
                      )}
                    >
                      {/* Aging Indicator */}
                      {patient.wait_minutes > 0 && (
                        <div className="absolute top-0 left-0 h-1 bg-blue-500/30" style={{ width: `${Math.min(patient.wait_minutes * 10, 100)}%` }} />
                      )}

                      <div className="flex items-start justify-between gap-4">
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-3">
                            <div className={cn(
                              "w-10 h-10 rounded-xl flex items-center justify-center text-lg font-black shadow-inner",
                              getESIColor(patient.esi_level)
                            )}>
                              {patient.esi_level}
                            </div>
                            <div>
                              <h3 className="font-black text-white uppercase tracking-tight">Patient {patient.patient_id}</h3>
                              <div className="flex items-center gap-2 text-[10px] font-bold text-slate-500">
                                <span>{patient.age}Y • {patient.gender}</span>
                                <span>•</span>
                                <span className={cn(patient.wait_minutes >= 3 ? "text-blue-400" : "")}>
                                  WAIT: {patient.wait_minutes * 30} MIN
                                </span>
                                {patient.effective_esi < patient.esi_level && (
                                  <span className="text-yellow-500 flex items-center gap-1">
                                    <Zap className="w-2.5 h-2.5" /> AGED UP
                                  </span>
                                )}
                              </div>
                            </div>
                          </div>
                          
                          <p className="text-sm text-slate-400 font-medium line-clamp-2 mb-3">
                            {patient.primary_symptoms.join(', ')}
                          </p>

                          <div className="flex flex-wrap gap-2">
                            {patient.required_resources.slice(0, 2).map((r, i) => (
                              <span key={i} className="text-[9px] font-black px-2 py-0.5 bg-slate-900 text-slate-500 rounded border border-slate-800">
                                {r}
                              </span>
                            ))}
                          </div>
                        </div>

                        {/* IoT Vitals Panel */}
                        <div className="w-24 space-y-2">
                          <div className={cn(
                            "p-2 rounded-xl border flex flex-col items-center justify-center gap-1",
                            patient.vitals?.status === 'critical' ? "bg-red-600 border-red-400" : "bg-slate-900 border-slate-800"
                          )}>
                            <Heart className={cn("w-4 h-4", patient.vitals?.status === 'critical' ? "text-white animate-ping" : "text-red-500")} />
                            <span className="text-xs font-black">{Math.round(patient.vitals?.hr || 0)}</span>
                            <span className="text-[8px] font-bold opacity-50">BPM</span>
                          </div>
                          <div className={cn(
                            "p-2 rounded-xl border flex flex-col items-center justify-center gap-1",
                            (patient.vitals?.spo2 || 100) < 90 ? "bg-red-600 border-red-400" : "bg-slate-900 border-slate-800"
                          )}>
                            <Activity className={cn("w-4 h-4", (patient.vitals?.spo2 || 100) < 90 ? "text-white" : "text-blue-400")} />
                            <span className="text-xs font-black">{Math.round(patient.vitals?.spo2 || 0)}%</span>
                            <span className="text-[8px] font-bold opacity-50">SpO2</span>
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </AnimatePresence>
              )}
            </div>
          </section>
        </div>
      </main>

      {/* Footer / Status Bar */}
      <footer className="fixed bottom-0 w-full bg-slate-900/90 backdrop-blur-md border-t border-slate-800 px-6 py-2 flex items-center justify-between text-[10px] font-bold text-slate-500 uppercase tracking-widest">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" />
            AI Engine Online
          </span>
          <span className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 bg-blue-500 rounded-full" />
            IoT Stream Active
          </span>
        </div>
        <p>© 2026 ER TRIAGE PRO • ADVANCED CLINICAL DECISION SUPPORT</p>
      </footer>
    </div>
  );
}
