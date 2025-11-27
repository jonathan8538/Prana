import React, { useState, useEffect } from 'react';
import { Play, Pause, Sparkles, Trophy, Target, ChevronRight, Star, Award, Activity, TrendingUp, CheckCircle, Users, Zap, Video, Github, Linkedin, Mail } from 'lucide-react';
import io from 'socket.io-client';
import megaMendung from './assets/mega-mendung.png';

const SOCKET_URL = 'https://dawn-unreversible-dreama.ngrok-free.dev ';
// ✨ TAMBAH COMPONENT INI
const FloatingBatik = () => {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {/* Mega Mendung 1 - Top Left */}
      <div 
        className="absolute opacity-10"
        style={{
          top: '5%',
          left: '3%',
          width: '200px',
          height: '200px',
          backgroundImage: `url(${megaMendung})`,
          backgroundSize: 'contain',
          backgroundRepeat: 'no-repeat',
          filter: 'brightness(0.8) saturate(0.7)',
          animation: 'float1 20s ease-in-out infinite'
        }}
      />

      {/* Mega Mendung 2 - Top Right */}
      <div 
        className="absolute opacity-8"
        style={{
          top: '10%',
          right: '5%',
          width: '250px',
          height: '250px',
          backgroundImage: `url(${megaMendung})`,
          backgroundSize: 'contain',
          backgroundRepeat: 'no-repeat',
          filter: 'brightness(0.8) saturate(0.7)',
          animation: 'float2 25s ease-in-out infinite'
        }}
      />

      {/* Mega Mendung 3 - Middle Left */}
      <div 
        className="absolute opacity-12"
        style={{
          top: '40%',
          left: '10%',
          width: '180px',
          height: '180px',
          backgroundImage: `url(${megaMendung})`,
          backgroundSize: 'contain',
          backgroundRepeat: 'no-repeat',
          filter: 'brightness(0.8) saturate(0.7)',
          animation: 'float3 30s ease-in-out infinite'
        }}
      />

      {/* Mega Mendung 4 - Middle Right */}
      <div 
        className="absolute opacity-10"
        style={{
          top: '50%',
          right: '8%',
          width: '220px',
          height: '220px',
          backgroundImage: `url(${megaMendung})`,
          backgroundSize: 'contain',
          backgroundRepeat: 'no-repeat',
          filter: 'brightness(0.8) saturate(0.7)',
          animation: 'float4 22s ease-in-out infinite'
        }}
      />

      {/* Mega Mendung 5 - Bottom Left */}
      <div 
        className="absolute opacity-8"
        style={{
          bottom: '15%',
          left: '15%',
          width: '190px',
          height: '190px',
          backgroundImage: `url(${megaMendung})`,
          backgroundSize: 'contain',
          backgroundRepeat: 'no-repeat',
          filter: 'brightness(0.8) saturate(0.7)',
          animation: 'float5 28s ease-in-out infinite'
        }}
      />

      {/* Mega Mendung 6 - Bottom Right */}
      <div 
        className="absolute opacity-10"
        style={{
          bottom: '8%',
          right: '12%',
          width: '210px',
          height: '210px',
          backgroundImage: `url(${megaMendung})`,
          backgroundSize: 'contain',
          backgroundRepeat: 'no-repeat',
          filter: 'brightness(0.8) saturate(0.7)',
          animation: 'float1 24s ease-in-out infinite'
        }}
      />
    </div>
  );
};
function App() {
  const [socket, setSocket] = useState(null);
  const [currentView, setCurrentView] = useState('landing');
  const [selectedLevel, setSelectedLevel] = useState(0);
  const [isConnected, setIsConnected] = useState(false);

  const [webcamFrame, setWebcamFrame] = useState(null);
  const [tutorialFrame, setTutorialFrame] = useState(null);
  const [currentPose, setCurrentPose] = useState('--');
  const [confidence, setConfidence] = useState(0);
  const [status, setStatus] = useState('READY');
  const [accuracy, setAccuracy] = useState(0);
  const [combo, setCombo] = useState(0);
  const [score, setScore] = useState({ correct: 0, total: 0 });
  const [targetPose, setTargetPose] = useState('Agem_Kanan');

  const danceMovements = [
    {
      id: 1,
      name: 'Agem Kanan',
      description: 'Posisi dasar tari Bali Mengayun 2 Tangan',
      icon: '🙏',
      difficulty: 'Beginner',
      duration: '5-10 min',
      color: 'from-blue-500 to-cyan-500',
      bgGlow: 'bg-blue-500/20',
      borderColor: 'border-blue-500/50'
    },
    {
      id: 2,
      name: 'Ngeed',
      description: 'Posisi dasar tari Bali bernama Ngeed',
      icon: '💃',
      difficulty: 'Intermediate',
      duration: '10-15 min',
      color: 'from-purple-500 to-pink-500',
      bgGlow: 'bg-purple-500/20',
      borderColor: 'border-purple-500/50'
    },
    {
      id: 3,
      name: 'Ngegol',
      description: 'Gerakan dasar tari bali Ngegol',
      icon: '✨',
      difficulty: 'Advanced',
      duration: '15-20 min',
      color: 'from-orange-500 to-red-500',
      bgGlow: 'bg-orange-500/20',
      borderColor: 'border-orange-500/50'
    }
  ];

  const features = [
    { icon: <Activity className="w-6 h-6" />, text: 'Real-time AI Detection', color: 'text-blue-400' },
    { icon: <Trophy className="w-6 h-6" />, text: 'Gamified Learning', color: 'text-yellow-400' },
    { icon: <TrendingUp className="w-6 h-6" />, text: 'Progress Tracking', color: 'text-green-400' },
    { icon: <Star className="w-6 h-6" />, text: 'Instant Feedback', color: 'text-purple-400' }
  ];

  const stats = [
    { value: '1000+', label: 'Students', icon: '👥' },
    { value: '95%', label: 'Success Rate', icon: '📈' },
    { value: '3', label: 'Dance Moves', icon: '💃' },
    { value: '4.9', label: 'Rating', icon: '⭐' }
  ];

  const howItWorks = [
    {
      step: '01',
      title: 'Choose Your Movement',
      description: 'Select from beginner to advanced Balinese dance movements',
      icon: <Target className="w-8 h-8" />,
      color: 'from-blue-500 to-cyan-500'
    },
    {
      step: '02',
      title: 'Follow the Tutorial',
      description: 'Watch and mirror the professional dancer in real-time video',
      icon: <Video className="w-8 h-8" />,
      color: 'from-purple-500 to-pink-500'
    },
    {
      step: '03',
      title: 'Get AI Feedback',
      description: 'Receive instant pose detection and accuracy scores',
      icon: <Zap className="w-8 h-8" />,
      color: 'from-orange-500 to-red-500'
    },
    {
      step: '04',
      title: 'Master & Progress',
      description: 'Track your improvement and unlock new movements',
      icon: <Trophy className="w-8 h-8" />,
      color: 'from-green-500 to-emerald-500'
    }
  ];

  const testimonials = [
    {
      name: 'Kadek Sari',
      role: 'Dance Student',
      avatar: '👩',
      text: 'PRANA helped me learn Balinese dance from home! The AI feedback is incredibly accurate and helpful.',
      rating: 5
    },
    {
      name: 'Made Wirawan',
      role: 'Dance Instructor',
      avatar: '👨',
      text: 'As a teacher, I recommend PRANA to all my students. The gamification keeps them motivated!',
      rating: 5
    },
    {
      name: 'Putu Ayu',
      role: 'Cultural Enthusiast',
      avatar: '👧',
      text: 'Finally, a modern way to preserve traditional dance! PRANA makes learning fun and accessible.',
      rating: 5
    }
  ];

  const benefits = [
    { icon: <CheckCircle className="w-5 h-5" />, text: 'Learn at your own pace, anytime, anywhere' },
    { icon: <CheckCircle className="w-5 h-5" />, text: 'No expensive dance classes required' },
    { icon: <CheckCircle className="w-5 h-5" />, text: 'Perfect for beginners to advanced dancers' },
    { icon: <CheckCircle className="w-5 h-5" />, text: 'Preserve and promote traditional culture' },
    { icon: <CheckCircle className="w-5 h-5" />, text: 'Track your progress with detailed analytics' },
    { icon: <CheckCircle className="w-5 h-5" />, text: 'Instant AI-powered pose correction' }
  ];

  useEffect(() => {
    const newSocket = io(SOCKET_URL);
    setSocket(newSocket);

    newSocket.on('connect', () => {
      console.log('✅ Connected to backend');
      setIsConnected(true);
    });

    newSocket.on('disconnect', () => {
      console.log('❌ Disconnected from backend');
      setIsConnected(false);
    });

    newSocket.on('frame_update', (data) => {
      setWebcamFrame(`data:image/jpeg;base64,${data.frame}`);
      setTutorialFrame(data.tutorial_frame ? `data:image/jpeg;base64,${data.tutorial_frame}` : null);
      setCurrentPose(data.current_pose || '--');
      setConfidence(data.confidence || 0);
      setStatus(data.status || 'READY');
      setAccuracy(data.accuracy || 0);
      setCombo(data.combo || 0);
      setScore(data.score || { correct: 0, total: 0 });
      setTargetPose(data.target || 'Agem_Kanan');
    });

    return () => newSocket.close();
  }, []);

  const startPractice = (levelId) => {
    if (socket && isConnected) {
      setSelectedLevel(levelId);
      setCurrentView('game');
      socket.emit('start_practice', { level: levelId });
    } else {
      alert('⚠️ Backend not connected! Please make sure Flask server is running.');
    }
  };

  const stopPractice = () => {
    if (socket) {
      socket.emit('stop_practice');
      setCurrentView('landing');
    }
  };

  const resetPractice = () => {
    if (socket) socket.emit('reset_practice');
  };

  if (currentView === 'landing') {
    return (
      <div className="min-h-screen bg-black text-white relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-purple-900/30 via-blue-900/30 to-pink-900/30" />
        <div className="absolute inset-0 opacity-30" style={{
          backgroundImage: 'radial-gradient(circle at 20% 30%, rgba(120, 119, 198, 0.4) 0%, transparent 50%), radial-gradient(circle at 80% 70%, rgba(236, 72, 153, 0.4) 0%, transparent 50%)',
        }} />

        <FloatingBatik />

        <div className="relative z-10">
          {/* Navigation */}
          <nav className="flex items-center justify-between px-12 py-6">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 bg-white rounded-xl flex items-center justify-center shadow-lg shadow-purple-500/50 p-1">
                <img src="/src/assets/LOGO PRANA.png" alt="Prana Logo" className="w-full h-full object-contain" />
              </div>
              <span className="text-2xl font-black bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                PRANA
              </span>
            </div>
            
            <div className="flex items-center gap-6">
              <a href="#features" className="text-gray-300 hover:text-white transition-colors">Features</a>
              <a href="#how-it-works" className="text-gray-300 hover:text-white transition-colors">How It Works</a>
              <a href="#testimonials" className="text-gray-300 hover:text-white transition-colors">Testimonials</a>
              <div className={`px-4 py-2 rounded-full text-sm font-semibold ${
                isConnected ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
              }`}>
                {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
              </div>
            </div>
          </nav>

          {/* Hero Section */}
          <div className="px-12 py-20 text-center max-w-6xl mx-auto">
            <div className="inline-flex items-center gap-2 bg-purple-500/20 border border-purple-500/50 rounded-full px-6 py-2 mb-8">
              <Sparkles className="w-4 h-4 text-purple-400" />
              <span className="text-sm font-semibold text-purple-300">AI-Powered Dance Learning Platform</span>
            </div>

            <h1 className="text-7xl font-black mb-6 leading-tight">
              Master <span className="bg-gradient-to-r from-purple-400 via-pink-400 to-orange-400 bg-clip-text text-transparent">
                Traditional Dance
              </span> Pose
            </h1>

            <p className="text-xl text-gray-400 mb-8 max-w-3xl mx-auto leading-relaxed">
              Learn authentic traditional dance pose with real-time AI pose detection. 
              Get instant feedback, track your progress, and master traditional movements at your own pace.
            </p>

            {/* CTA Buttons */}
            <div className="flex gap-4 justify-center mb-16">
              <button 
                onClick={() => document.getElementById('movements').scrollIntoView({ behavior: 'smooth' })}
                className="group bg-gradient-to-r from-purple-600 via-pink-600 to-orange-600 px-8 py-4 rounded-2xl font-bold text-lg hover:shadow-2xl hover:shadow-purple-500/50 transition-all transform hover:scale-105 flex items-center gap-3"
              >
                <Play className="w-6 h-6" />
                Start Learning Free
                <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </button>
              
              <button className="bg-white/10 backdrop-blur-xl border border-white/20 px-8 py-4 rounded-2xl font-bold text-lg hover:bg-white/20 transition-all">
                <Video className="w-6 h-6 inline mr-2" />
                Watch Demo
              </button>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-4 gap-6 max-w-4xl mx-auto">
              {stats.map((stat, idx) => (
                <div key={idx} className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6 hover:bg-white/10 transition-all transform hover:scale-105">
                  <div className="text-4xl mb-2">{stat.icon}</div>
                  <div className="text-3xl font-black bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent mb-1">
                    {stat.value}
                  </div>
                  <div className="text-sm text-gray-400">{stat.label}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Benefits Section */}
          <div className="px-12 py-20 bg-gradient-to-b from-transparent via-purple-900/10 to-transparent">
            <div className="max-w-6xl mx-auto">
              <div className="text-center mb-12">
                <h2 className="text-4xl font-black mb-4">
                  Why <span className="bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">PRANA</span> is Different
                </h2>
                <p className="text-gray-400 text-lg">Everything you need to master traditional dance</p>
              </div>

              <div className="grid grid-cols-2 gap-6 max-w-4xl mx-auto">
                {benefits.map((benefit, idx) => (
                  <div key={idx} className="flex items-start gap-3 bg-white/5 backdrop-blur-xl border border-white/10 rounded-xl p-4 hover:bg-white/10 transition-all">
                    <div className="text-green-400 mt-1">{benefit.icon}</div>
                    <p className="text-gray-300">{benefit.text}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* How It Works Section */}
          <div id="how-it-works" className="px-12 py-20 bg-gradient-to-b from-transparent to-purple-900/20">
            <div className="max-w-6xl mx-auto">
              <div className="text-center mb-16">
                <h2 className="text-4xl font-black mb-4">
                  How <span className="bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">It Works</span>
                </h2>
                <p className="text-gray-400 text-lg">Start learning in 4 simple steps</p>
              </div>

              <div className="grid grid-cols-4 gap-8">
                {howItWorks.map((item, idx) => (
                  <div key={idx} className="relative-h-full">
                    <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6 hover:bg-white/10 transition-all group h-full flex flex-col">
                      <div className={`w-16 h-16 bg-gradient-to-br ${item.color} rounded-2xl flex items-center justify-center mb-4 text-white shadow-lg`}>
                        {item.icon}
                      </div>
                      <div className={`text-5xl font-black mb-2 bg-gradient-to-br ${item.color} bg-clip-text text-transparent`}>
                      {item.step}
                      </div>
                      <h3 className="text-xl font-bold mb-2">{item.title}</h3>
                      <p className="text-sm text-gray-400 leading-relaxed">{item.description}</p>
                    </div>
                    {idx < howItWorks.length - 1 && (
                      <div className="hidden lg:block absolute top-1/2 -right-4 w-8 h-0.5 bg-gradient-to-r from-purple-500 to-pink-500" />
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Features Section */}
          <div id="features" className="px-12 py-20">
            <div className="max-w-6xl mx-auto">
              <h2 className="text-4xl font-black text-center mb-12">
                Powerful <span className="bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">Features</span>
              </h2>

              <div className="grid grid-cols-4 gap-6">
                {features.map((feature, idx) => (
                  <div key={idx} className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6 hover:bg-white/10 transition-all group transform hover:scale-105">
                    <div className={`${feature.color} mb-4 transform group-hover:scale-110 transition-transform`}>
                      {feature.icon}
                    </div>
                    <p className="font-semibold">{feature.text}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Dance Movements Section */}
          <div id="movements" className="px-12 py-20 bg-gradient-to-b from-purple-900/20 to-transparent">
            <div className="max-w-6xl mx-auto">
              <h2 className="text-4xl font-black text-center mb-4">
                Choose Your <span className="bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">Dance Movement</span>
              </h2>
              <p className="text-center text-gray-400 mb-12 text-lg">Start with beginner moves and progress to advanced techniques</p>

              <div className="grid grid-cols-3 gap-8">
                {danceMovements.map((movement, idx) => (
                  <div 
                    key={movement.id}
                    className={`group relative bg-gradient-to-br ${movement.bgGlow} backdrop-blur-xl border ${movement.borderColor} rounded-3xl p-8 hover:scale-105 transition-all cursor-pointer overflow-hidden`}
                    onClick={() => startPractice(idx)}
                  >
                    <div className={`absolute inset-0 bg-gradient-to-br ${movement.color} opacity-0 group-hover:opacity-20 transition-opacity`} />
                    
                    <div className="relative">
                      <div className={`w-20 h-20 bg-gradient-to-br ${movement.color} rounded-2xl flex items-center justify-center text-4xl mb-6 shadow-xl`}>
                        {movement.icon}
                      </div>

                      <h3 className="text-2xl font-black mb-3">{movement.name}</h3>
                      <p className="text-gray-400 text-sm mb-6 leading-relaxed">{movement.description}</p>

                      <div className="flex items-center justify-between text-sm mb-6">
                        <span className={`bg-gradient-to-r ${movement.color} bg-clip-text text-transparent font-bold`}>
                          {movement.difficulty}
                        </span>
                        <span className="text-gray-400">⏱️ {movement.duration}</span>
                      </div>

                      <button className={`w-full bg-gradient-to-r ${movement.color} py-3 rounded-xl font-bold hover:shadow-lg transition-all flex items-center justify-center gap-2`}>
                        Start Learning
                        <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Testimonials Section */}
          <div id="testimonials" className="px-12 py-20">
            <div className="max-w-6xl mx-auto">
              <div className="text-center mb-12">
                <h2 className="text-4xl font-black mb-4">
                  What Our <span className="bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">Students Say</span>
                </h2>
                <p className="text-gray-400 text-lg">Join thousands of happy learners</p>
              </div>

              <div className="grid grid-cols-3 gap-8">
                {testimonials.map((testimonial, idx) => (
                  <div key={idx} className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6 hover:bg-white/10 transition-all">
                    <div className="flex items-center gap-4 mb-4">
                      <div className="text-5xl">{testimonial.avatar}</div>
                      <div>
                        <h4 className="font-bold">{testimonial.name}</h4>
                        <p className="text-sm text-gray-400">{testimonial.role}</p>
                      </div>
                    </div>
                    <div className="flex gap-1 mb-3">
                      {[...Array(testimonial.rating)].map((_, i) => (
                        <Star key={i} className="w-4 h-4 text-yellow-400 fill-yellow-400" />
                      ))}
                    </div>
                    <p className="text-gray-300 text-sm leading-relaxed italic">"{testimonial.text}"</p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* CTA Section */}
          <div className="px-12 py-20">
            <div className="max-w-4xl mx-auto bg-gradient-to-r from-purple-600 via-pink-600 to-orange-600 rounded-3xl p-12 text-center shadow-2xl">
              <Trophy className="w-16 h-16 mx-auto mb-6 text-white" />
              <h2 className="text-4xl font-black mb-4">Ready to Start Your Dance Journey?</h2>
              <p className="text-lg mb-8 text-white/90">Join thousands of students learning traditional dance pose with AI</p>
              <div className="flex gap-4 justify-center">
                <button 
                  onClick={() => document.getElementById('movements').scrollIntoView({ behavior: 'smooth' })}
                  className="bg-white text-purple-600 px-8 py-4 rounded-2xl font-bold text-lg hover:bg-gray-100 transition-all transform hover:scale-105 inline-flex items-center gap-3"
                >
                  <Play className="w-6 h-6" />
                  Start Learning Now - It's Free!
                </button>
              </div>
            </div>
          </div>

          {/* Footer */}
          <footer className="px-12 py-12 border-t border-white/10 bg-black/50">
            <div className="max-w-6xl mx-auto">
              <div className="grid grid-cols-4 gap-8 mb-8">
                <div>
                  <div className="flex items-center gap-2 mb-4">
                    <div className="w-10 h-10 bg-white rounded-lg flex items-center justify-center p-1">
                      <img src="/src/assets/LOGO PRANA.png" alt="Prana Logo" className="w-full h-full object-contain" />
                    </div>
                    <span className="text-xl font-black bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                      PRANA
                    </span>
                  </div>
                  <p className="text-sm text-gray-400">
                    AI-powered platform for learning traditional dance pose
                  </p>
                </div>

                <div>
                  <h4 className="font-bold mb-4">Product</h4>
                  <ul className="space-y-2 text-sm text-gray-400">
                    <li><a href="#" className="hover:text-white transition-colors">Features</a></li>
                    <li><a href="#" className="hover:text-white transition-colors">How It Works</a></li>
                    <li><a href="#" className="hover:text-white transition-colors">Pricing</a></li>
                    <li><a href="#" className="hover:text-white transition-colors">FAQ</a></li>
                  </ul>
                </div>

                <div>
                  <h4 className="font-bold mb-4">Company</h4>
                  <ul className="space-y-2 text-sm text-gray-400">
                    <li><a href="#" className="hover:text-white transition-colors">About Us</a></li>
                    <li><a href="#" className="hover:text-white transition-colors">Blog</a></li>
                    <li><a href="#" className="hover:text-white transition-colors">Careers</a></li>
                    <li><a href="#" className="hover:text-white transition-colors">Contact</a></li>
                  </ul>
                </div>

                <div>
                  <h4 className="font-bold mb-4">Connect</h4>
                  <div className="flex gap-3">
                    <a href="#" className="w-10 h-10 bg-white/10 hover:bg-white/20 rounded-lg flex items-center justify-center transition-colors">
                      <Github className="w-5 h-5" />
                    </a>
                    <a href="#" className="w-10 h-10 bg-white/10 hover:bg-white/20 rounded-lg flex items-center justify-center transition-colors">
                      <Linkedin className="w-5 h-5" />
                    </a>
                    <a href="#" className="w-10 h-10 bg-white/10 hover:bg-white/20 rounded-lg flex items-center justify-center transition-colors">
                      <Mail className="w-5 h-5" />
                    </a>
                  </div>
                </div>
              </div>

              <div className="border-t border-white/10 pt-8 flex items-center justify-between text-sm text-gray-400">
                <p>© 2025 PRANA. All rights reserved.</p>
                <div className="flex gap-6">
                  <a href="#" className="hover:text-white transition-colors">Privacy Policy</a>
                  <a href="#" className="hover:text-white transition-colors">Terms of Service</a>
                  <a href="#" className="hover:text-white transition-colors">Cookie Policy</a>
                </div>
              </div>
            </div>
          </footer>
        </div>
      </div>
    );
  }

  // Practice Mode (unchanged)
  return (
    <div className="min-h-screen bg-black text-white relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-br from-purple-900/20 via-blue-900/20 to-pink-900/20" />
      
      <div className="relative z-10 p-8">
        <button 
          onClick={stopPractice}
          className="mb-6 bg-white/10 backdrop-blur-xl border border-white/20 px-6 py-3 rounded-xl font-semibold hover:bg-white/20 transition-all"
        >
          ← Back to Home
        </button>

        <div className="text-center mb-8">
          <h1 className="text-5xl font-black bg-gradient-to-r from-purple-400 via-pink-400 to-orange-400 bg-clip-text text-transparent mb-2">
            PRACTICE MODE
          </h1>
          <p className="text-gray-400">Level {selectedLevel + 1}: {danceMovements[selectedLevel].name}</p>
        </div>

        <div className="max-w-7xl mx-auto">
          <div className="bg-gradient-to-br from-gray-900/50 to-gray-800/50 backdrop-blur-xl rounded-3xl overflow-hidden border border-gray-700/50 shadow-2xl">
            
            <div className="grid grid-cols-2 aspect-video">
              <div className="relative bg-gray-900 border-r border-gray-700">
                <div className="absolute top-4 left-4 bg-red-500 px-4 py-2 rounded-full text-sm font-bold flex items-center gap-2 z-10">
                  <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                  LIVE
                </div>
                
                {webcamFrame ? (
                  <img src={webcamFrame} alt="Webcam" className="w-full h-full object-cover" />
                ) : (
                  <div className="h-full flex items-center justify-center">
                    <div className="text-center">
                      <Activity className="w-16 h-16 text-purple-400 mx-auto mb-4" />
                      <p className="text-gray-400">Waiting for camera...</p>
                    </div>
                  </div>
                )}

                <div className="absolute bottom-4 left-4 right-4 bg-black/80 backdrop-blur-sm rounded-2xl p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs text-gray-400">Detected Pose</span>
                    <span className={`text-xs font-bold px-3 py-1 rounded-full ${
                      status === 'PERFECT!' ? 'bg-green-500/20 text-green-400' :
                      status === 'GOOD' ? 'bg-yellow-500/20 text-yellow-400' :
                      'bg-red-500/20 text-red-400'
                    }`}>
                      {status}
                    </span>
                  </div>
                  <p className="text-lg font-bold mb-2">{currentPose}</p>
                  <div className="flex items-center gap-2">
                    <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-green-500 to-emerald-500 transition-all"
                        style={{ width: `${confidence}%` }}
                      />
                    </div>
                    <span className="text-sm font-bold text-green-400">{confidence.toFixed(0)}%</span>
                  </div>
                </div>
              </div>

              <div className="relative bg-gradient-to-br from-purple-900 to-pink-900 flex items-center justify-center overflow-hidden">
                <div className="absolute top-4 left-4 bg-yellow-400 text-gray-900 px-4 py-2 rounded-full text-sm font-bold z-10">
                  ✨ Tutorial
                </div>
                
                {tutorialFrame ? (
                  <img src={tutorialFrame} alt="Tutorial" className="w-full h-full object-cover" />
                ) : (
                  <div className="text-center">
                    <div className="w-32 h-32 bg-white/10 rounded-full flex items-center justify-center mb-4 border-4 border-white/20">
                      <span className="text-6xl">{danceMovements[selectedLevel].icon}</span>
                    </div>
                    <p className="text-xl font-bold">{danceMovements[selectedLevel].name}</p>
                    <p className="text-sm text-gray-300 mt-2">Loading tutorial...</p>
                  </div>
                )}
              </div>
            </div>

            <div className="bg-gray-900/80 px-8 py-6 grid grid-cols-3 gap-6">
              <div>
                <p className="text-sm text-gray-400 mb-2">Accuracy</p>
                <div className="flex items-center gap-3">
                  <div className={`text-3xl font-black ${accuracy >= 70 ? 'text-green-400' : 'text-orange-400'}`}>
                    {accuracy.toFixed(1)}%
                  </div>
                  <div className="flex-1 h-3 bg-gray-800 rounded-full overflow-hidden">
                    <div 
                      className={`h-full transition-all ${
                        accuracy >= 70 ? 'bg-gradient-to-r from-green-500 to-emerald-500' : 'bg-gradient-to-r from-orange-500 to-red-500'
                      }`}
                      style={{ width: `${Math.min(100, accuracy)}%` }}
                    />
                  </div>
                </div>
              </div>

              <div>
                <p className="text-sm text-gray-400 mb-2">Combo Streak</p>
                <div className="flex items-center gap-2">
                  <div className="text-3xl font-black bg-gradient-to-r from-yellow-400 to-orange-400 bg-clip-text text-transparent">
                    {combo}x
                  </div>
                  <div className="flex gap-1">
                    {[...Array(Math.min(5, Math.floor(combo / 20)))].map((_, i) => (
                      <Star key={i} className="w-4 h-4 text-yellow-400 fill-yellow-400" />
                    ))}
                  </div>
                </div>
              </div>

              <div>
                <p className="text-sm text-gray-400 mb-2">Score</p>
                <div className="text-3xl font-black text-white">
                  {score.correct} / {score.total}
                </div>
                <p className="text-xs text-gray-500 mt-1">frames processed</p>
              </div>
            </div>

            <div className="bg-gray-950/80 px-8 py-4">
              <div className="flex items-center justify-between text-xs mb-2">
                <span className="text-gray-400">Session Progress</span>
                <span className="text-gray-400">{score.total} frames</span>
              </div>
              <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-purple-500 via-pink-500 to-orange-500 transition-all"
                  style={{ width: `${Math.min(100, (score.total / 500) * 100)}%` }}
                />
              </div>
            </div>
          </div>

          <div className="mt-8 flex justify-center gap-4">
            <button 
              onClick={stopPractice}
              className="bg-red-600 hover:bg-red-700 px-8 py-4 rounded-2xl font-bold transition-all flex items-center gap-2"
            >
              <Pause className="w-5 h-5" />
              Stop Practice
            </button>
            
            <button 
              onClick={resetPractice}
              className="bg-gray-700 hover:bg-gray-600 px-8 py-4 rounded-2xl font-bold transition-all"
            >
              Reset Stats
            </button>
          </div>

          <div className="mt-8 max-w-4xl mx-auto bg-blue-500/10 backdrop-blur-xl border border-blue-500/30 rounded-2xl p-6">
            <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
              <Target className="w-5 h-5 text-blue-400" />
              Tips for Better Accuracy
            </h3>
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div className="bg-black/30 rounded-xl p-3">
                <p className="text-blue-400 font-semibold mb-1">👁️ Camera Position</p>
                <p className="text-gray-400">Keep your full body visible in frame</p>
              </div>
              <div className="bg-black/30 rounded-xl p-3">
                <p className="text-green-400 font-semibold mb-1">⚡ Hold Poses</p>
                <p className="text-gray-400">Maintain each pose for 2-3 seconds</p>
              </div>
              <div className="bg-black/30 rounded-xl p-3">
                <p className="text-purple-400 font-semibold mb-1">🎯 Follow Tutorial</p>
                <p className="text-gray-400">Mirror the reference movement</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;