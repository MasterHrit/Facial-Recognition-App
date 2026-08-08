// BioScan AI - Core Frontend Logic

document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('overlay-canvas');
    const systemStatusDot = document.getElementById('system-status-dot');
    const systemStatusText = document.getElementById('system-status-text');
    
    const regNameInput = document.getElementById('reg-name');
    const btnRegister = document.getElementById('btn-register');
    const btnPauseCamera = document.getElementById('btn-pause-camera');
    const btnClearLogs = document.getElementById('btn-clear-logs');
    
    const userTableBody = document.getElementById('user-table-body');
    const logsList = document.getElementById('logs-list');
    
    // Stats Elements
    const statWebcam = document.getElementById('stat-webcam');
    const statRegisteredCount = document.getElementById('stat-registered-count');
    const statDetectionsCount = document.getElementById('stat-detections-count');
    const statLastMatch = document.getElementById('stat-last-match');
    const dbProfileBadge = document.getElementById('db-profile-badge');
    const cameraFallback = document.getElementById('camera-fallback');
    const fallbackText = document.getElementById('fallback-text');
    const cameraCardContainer = document.getElementById('camera-card-container');
    
    // State Variables
    let modelsLoaded = false;
    let localStream = null;
    let detectionInterval = null;
    let activeFaceDescriptor = null;
    let lastDetectionResult = null;
    
    // Rate-limiting for recognition logging
    let lastLoggedName = "";
    let lastLoggedTime = 0;
    const LOG_COOLDOWN_MS = 8000; // Log same person every 8s
    const UNKNOWN_LOG_COOLDOWN_MS = 15000; // Log unknown every 15s
    
    // Total detection counts for today (reset on page load, or counts log list)
    let detectionsToday = 0;

    // Initialize Application
    async function init() {
        try {
            updateSystemStatus('loading', 'Loading AI Models...');
            
            // Load models from Flask static directory
            const MODEL_URL = '/static/models';
            await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
            await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
            await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
            
            modelsLoaded = true;
            fallbackText.textContent = "Requesting Webcam access...";
            updateSystemStatus('loading', 'Starting Camera...');
            
            // Load Database contents
            await loadProfiles();
            await loadLogs();
            
            // Start Camera
            await startWebcam();
            
        } catch (error) {
            console.error("Initialization error:", error);
            updateSystemStatus('error', 'Initialization Failed');
            fallbackText.innerHTML = `<span style="color: var(--danger)"><i class="fa-solid fa-triangle-exclamation"></i> Model loading failed. Ensure weights exist in /static/models/.</span>`;
        }
    }

    // Update Status Badge
    function updateSystemStatus(state, text) {
        systemStatusDot.className = 'status-dot';
        if (state === 'active') {
            systemStatusDot.classList.add('active');
            systemStatusText.style.color = 'var(--text-primary)';
        } else if (state === 'loading') {
            systemStatusDot.classList.add('loading');
            systemStatusText.style.color = '#f59e0b';
        } else if (state === 'error') {
            systemStatusDot.style.backgroundColor = 'var(--danger)';
            systemStatusDot.style.boxShadow = '0 0 10px var(--danger)';
            systemStatusText.style.color = 'var(--danger)';
        }
        systemStatusText.textContent = text;
    }

    // Start Webcam Feed
    async function startWebcam() {
        try {
            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                },
                audio: false
            };
            
            localStream = await navigator.mediaDevices.getUserMedia(constraints);
            video.srcObject = localStream;
            
            video.onloadedmetadata = () => {
                video.play();
                isStreaming = true;
                statWebcam.textContent = "Active";
                statWebcam.style.color = "var(--accent)";
                cameraFallback.style.display = "none";
                regNameInput.disabled = false;
                btnRegister.disabled = false;
                btnPauseCamera.disabled = false;
                
                updateSystemStatus('active', 'System Ready');
                
                // Align canvas resolution with video dimensions
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                
                startDetectionLoop();
            };
        } catch (err) {
            console.error("Webcam access denied:", err);
            statWebcam.textContent = "Error";
            statWebcam.style.color = "var(--danger)";
            fallbackText.innerHTML = `<span style="color: var(--danger)"><i class="fa-solid fa-video-slash"></i> Camera access denied. Enable camera permissions in browser settings.</span>`;
            updateSystemStatus('error', 'Camera Access Denied');
        }
    }

    // Detection & Recognition Loop
    function startDetectionLoop() {
        if (detectionInterval) clearInterval(detectionInterval);
        
        // Run detection every 200ms for responsiveness and performance
        detectionInterval = setInterval(async () => {
            if (video.paused || video.ended || !modelsLoaded) return;
            
            const displaySize = { width: video.videoWidth, height: video.videoHeight };
            
            // Detect faces
            const detections = await faceapi.detectAllFaces(
                video, 
                new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 })
            ).withFaceLandmarks().withFaceDescriptors();
            
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            if (detections.length === 0) {
                activeFaceDescriptor = null;
                lastDetectionResult = null;
                cameraCardContainer.classList.remove('scanning');
                return;
            }
            
            cameraCardContainer.classList.add('scanning');
            const resizedDetections = faceapi.resizeResults(detections, displaySize);
            
            // Capture primary face descriptor (first one detected)
            activeFaceDescriptor = detections[0].descriptor;
            
            // Query server to recognize this descriptor
            try {
                const response = await fetch('/api/recognize', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ embedding: Array.from(activeFaceDescriptor) })
                });
                
                if (response.ok) {
                    const result = await response.json();
                    lastDetectionResult = result;
                    
                    // Trigger Logging with Cooldown rate limit
                    handleLogEvent(result, detections[0].detection.box);
                }
            } catch (error) {
                console.error("Error recognizing face descriptor:", error);
            }
            
            // Custom Sci-Fi Draw Bounding Boxes and Landmarks
            resizedDetections.forEach((detection, index) => {
                const box = detection.detection.box;
                const isPrimary = (index === 0);
                
                // Color mapping: Purple/Blue for scanning, Green for recognized, Red for unknown
                let color = '#a855f7'; // Secondary purple
                let displayName = "Analyzing...";
                
                if (isPrimary && lastDetectionResult) {
                    if (lastDetectionResult.status === 'recognized') {
                        color = '#10b981'; // Accent green
                        displayName = `${lastDetectionResult.name} (${Math.round(lastDetectionResult.confidence * 100)}%)`;
                    } else if (lastDetectionResult.status === 'unknown') {
                        color = '#ef4444'; // Danger red
                        displayName = "Unknown Face";
                    }
                }
                
                // Corner Brackets
                ctx.strokeStyle = color;
                ctx.lineWidth = 3;
                ctx.shadowBlur = 8;
                ctx.shadowColor = color;
                
                const len = Math.min(18, box.width * 0.15);
                
                // Top Left
                ctx.beginPath(); ctx.moveTo(box.x, box.y + len); ctx.lineTo(box.x, box.y); ctx.lineTo(box.x + len, box.y); ctx.stroke();
                // Top Right
                ctx.beginPath(); ctx.moveTo(box.x + box.width - len, box.y); ctx.lineTo(box.x + box.width, box.y); ctx.lineTo(box.x + box.width, box.y + len); ctx.stroke();
                // Bottom Left
                ctx.beginPath(); ctx.moveTo(box.x, box.y + box.height - len); ctx.lineTo(box.x, box.y + box.height); ctx.lineTo(box.x + len, box.y + box.height); ctx.stroke();
                // Bottom Right
                ctx.beginPath(); ctx.moveTo(box.x + box.width - len, box.y + box.height); ctx.lineTo(box.x + box.width, box.y + box.height); ctx.lineTo(box.x + box.width, box.y + box.height - len); ctx.stroke();
                
                // Label box
                ctx.shadowBlur = 0;
                ctx.fillStyle = color;
                ctx.font = "bold 13px 'Inter', sans-serif";
                
                // Draw text above box
                const textWidth = ctx.measureText(displayName).width;
                ctx.fillRect(box.x, box.y - 25, textWidth + 16, 25);
                
                ctx.fillStyle = '#ffffff';
                ctx.fillText(displayName, box.x + 8, box.y - 8);
                
                // Draw Landmarks
                const landmarks = detection.landmarks;
                ctx.fillStyle = 'rgba(255, 255, 255, 0.45)';
                landmarks.positions.forEach(point => {
                    ctx.beginPath();
                    ctx.arc(point.x, point.y, 2, 0, 2 * Math.PI);
                    ctx.fill();
                });
            });
            
        }, 200);
    }

    // Capture Snapshot of detected face
    function captureFaceSnapshot(box) {
        const tempCanvas = document.createElement('canvas');
        const padding = 0.25; // add 25% padding
        
        let x = box.x - box.width * padding;
        let y = box.y - box.height * padding;
        let w = box.width * (1 + padding * 2);
        let h = box.height * (1 + padding * 2);
        
        // Mirror coordinates mapping for drawing mirrored canvas contents properly
        // The HTML video element is mirrored via scaleX(-1). We must capture from video.
        // Let's get mirrored X coordinate on the raw video element
        const rawVideoWidth = video.videoWidth;
        // Since webcam display is scaledX(-1), the crop box.x is from mirrored space.
        // Let's map it back to original canvas:
        // Actually, canvas crop is simpler if we crop directly from the mirrored video frame using a canvas.
        // If we draw raw video to a canvas, crop it, then mirror the cropped canvas!
        
        // Clamp bounds
        x = Math.max(0, x);
        y = Math.max(0, y);
        w = Math.min(video.videoWidth - x, w);
        h = Math.min(video.videoHeight - y, h);
        
        tempCanvas.width = 150;
        tempCanvas.height = 150;
        const ctx = tempCanvas.getContext('2d');
        
        // Mirror snapshot so it matches what user sees
        ctx.translate(150, 0);
        ctx.scale(-1, 1);
        
        ctx.drawImage(video, x, y, w, h, 0, 0, 150, 150);
        return tempCanvas.toDataURL('image/jpeg', 0.85);
    }

    // Handles triggering server-side DB log with snapshot upload
    async function handleLogEvent(result, box) {
        const now = Date.now();
        const name = result.name;
        const isUnknown = (result.status === 'unknown' || result.status === 'no_users');
        
        // Determine if cooldown applies
        const isDifferentPerson = (name !== lastLoggedName);
        const cooldown = isUnknown ? UNKNOWN_LOG_COOLDOWN_MS : LOG_COOLDOWN_MS;
        const hasCooldownExpired = (now - lastLoggedTime > cooldown);
        
        if (isDifferentPerson || hasCooldownExpired) {
            lastLoggedName = name;
            lastLoggedTime = now;
            
            // Capture base64 snapshot
            const snapshotB64 = captureFaceSnapshot(box);
            
            // Send log event to server
            try {
                const response = await fetch('/api/log_recognition', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name: name,
                        confidence: result.confidence,
                        image: snapshotB64
                    })
                });
                
                if (response.ok) {
                    // Update logs view
                    await loadLogs();
                    
                    // Update last match stats card
                    if (!isUnknown) {
                        statLastMatch.textContent = name;
                        statLastMatch.style.color = "var(--accent)";
                    } else {
                        statLastMatch.textContent = "Unknown Face";
                        statLastMatch.style.color = "var(--danger)";
                    }
                }
            } catch (err) {
                console.error("Failed to write activity log:", err);
            }
        }
    }

    // Register Face Action
    btnRegister.addEventListener('click', async () => {
        const name = regNameInput.value.trim();
        
        if (!name) {
            alert("Please enter a name for the profile.");
            return;
        }
        
        if (!activeFaceDescriptor) {
            alert("No face detected in webcam stream. Please position your face clearly in the camera center.");
            return;
        }
        
        btnRegister.disabled = true;
        regNameInput.disabled = true;
        updateSystemStatus('loading', `Registering ${name}...`);
        
        try {
            const response = await fetch('/api/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: name,
                    embedding: Array.from(activeFaceDescriptor)
                })
            });
            
            const result = await response.json();
            
            if (response.ok) {
                regNameInput.value = "";
                await loadProfiles();
                alert(`Successfully registered face for "${name}"!`);
            } else {
                alert(`Error: ${result.error || "Failed to register face"}`);
            }
        } catch (error) {
            console.error("Registration request failed:", error);
            alert("Network error. Failed to save profile template.");
        } finally {
            btnRegister.disabled = false;
            regNameInput.disabled = false;
            updateSystemStatus('active', 'System Ready');
        }
    });

    // Load Database User Profiles
    async function loadProfiles() {
        try {
            const response = await fetch('/api/users');
            if (response.ok) {
                const users = await response.json();
                
                // Update stats card & header badge
                statRegisteredCount.textContent = users.length;
                dbProfileBadge.textContent = `${users.length} Profile${users.length !== 1 ? 's' : ''}`;
                
                if (users.length === 0) {
                    userTableBody.innerHTML = `
                        <tr>
                            <td colspan="3" style="text-align: center; color: var(--text-secondary); padding: 2rem 0;">
                                No profiles registered yet. Enter a name above to create one.
                            </td>
                        </tr>`;
                    return;
                }
                
                userTableBody.innerHTML = "";
                users.forEach(user => {
                    const row = document.createElement('tr');
                    
                    const date = new Date(user.created_at);
                    const formattedDate = date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
                    
                    row.innerHTML = `
                        <td style="font-weight: 500;">${escapeHtml(user.name)}</td>
                        <td style="text-align: center;">
                            <span style="background: rgba(99, 102, 241, 0.15); color: var(--primary); padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">
                                ${user.embedding_count}
                            </span>
                        </td>
                        <td style="text-align: right;">
                            <button class="delete-btn" data-id="${user.id}" title="Delete Profile">
                                <i class="fa-solid fa-trash-can"></i>
                            </button>
                        </td>
                    `;
                    userTableBody.appendChild(row);
                });
                
                // Add Delete Listeners
                document.querySelectorAll('.delete-btn').forEach(btn => {
                    btn.addEventListener('click', async (e) => {
                        const id = e.currentTarget.getAttribute('data-id');
                        const row = e.currentTarget.closest('tr');
                        const name = row.cells[0].textContent;
                        
                        if (confirm(`Are you sure you want to delete profile "${name}"? This removes all their facial templates.`)) {
                            try {
                                const delResponse = await fetch(`/api/users/${id}`, { method: 'DELETE' });
                                if (delResponse.ok) {
                                    await loadProfiles();
                                } else {
                                    alert("Could not delete user template.");
                                }
                            } catch (err) {
                                console.error("Error deleting user:", err);
                            }
                        }
                    });
                });
            }
        } catch (error) {
            console.error("Failed to load users:", error);
        }
    }

    // Load Activity Logs Timeline
    async function loadLogs() {
        try {
            const response = await fetch('/api/logs');
            if (response.ok) {
                const logs = await response.json();
                
                // Calculate Detections count today
                // Filter logs that happened today
                const startOfDay = new Date();
                startOfDay.setHours(0, 0, 0, 0);
                const logsToday = logs.filter(log => new Date(log.timestamp) >= startOfDay);
                statDetectionsCount.textContent = logsToday.length;
                
                if (logs.length === 0) {
                    logsList.innerHTML = `
                        <div class="empty-logs">
                            <i class="fa-solid fa-fingerprint" style="font-size: 2rem; color: var(--text-secondary); margin-bottom: 0.5rem; display: block;"></i>
                            No recognition activity recorded yet.
                        </div>`;
                    return;
                }
                
                logsList.innerHTML = "";
                logs.forEach(log => {
                    const card = document.createElement('div');
                    card.className = 'log-card';
                    
                    const isUnknown = (log.name === 'Unknown');
                    const badgeClass = isUnknown ? 'unknown' : 'match';
                    const badgeText = isUnknown ? 'UNKNOWN' : 'MATCH';
                    
                    const time = new Date(log.timestamp);
                    const formattedTime = time.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' });
                    
                    // Display image snapshot if it exists, otherwise show placeholder initials
                    let avatarHtml = "";
                    if (log.snapshot_path) {
                        avatarHtml = `<img class="log-avatar" src="/static/${log.snapshot_path}" alt="Face Snapshot">`;
                    } else {
                        const initial = isUnknown ? "?" : log.name.charAt(0).toUpperCase();
                        avatarHtml = `<div class="log-avatar-placeholder">${initial}</div>`;
                    }
                    
                    card.innerHTML = `
                        <div class="log-left">
                            ${avatarHtml}
                            <div class="log-info">
                                <span class="log-name" style="${isUnknown ? 'color: var(--danger);' : ''}">${escapeHtml(log.name)}</span>
                                <span class="log-time">${formattedTime}</span>
                            </div>
                        </div>
                        <div class="log-right">
                            <span class="log-badge ${badgeClass}">${badgeText}</span>
                            <span class="log-confidence">${isUnknown ? '0%' : `${log.confidence}%`} confidence</span>
                        </div>
                    `;
                    logsList.appendChild(card);
                });
            }
        } catch (error) {
            console.error("Failed to load activity logs:", error);
        }
    }

    // Clear Recognition Logs Activity
    btnClearLogs.addEventListener('click', async () => {
        if (confirm("Are you sure you want to clear all recognition logs and delete saved snapshot files?")) {
            try {
                const response = await fetch('/api/clear_logs', { method: 'POST' });
                if (response.ok) {
                    lastLoggedName = "";
                    lastLoggedTime = 0;
                    statLastMatch.textContent = "None";
                    statLastMatch.style.color = "var(--text-secondary)";
                    await loadLogs();
                }
            } catch (err) {
                console.error("Error clearing activity logs:", err);
            }
        }
    });

    // Pause/Resume Camera
    btnPauseCamera.addEventListener('click', () => {
        if (video.paused) {
            video.play();
            btnPauseCamera.innerHTML = `<i class="fa-solid fa-pause"></i> Pause`;
            btnPauseCamera.className = "secondary-btn";
            cameraCardContainer.classList.add('scanning');
            statWebcam.textContent = "Active";
            statWebcam.style.color = "var(--accent)";
            updateSystemStatus('active', 'System Ready');
        } else {
            video.pause();
            btnPauseCamera.innerHTML = `<i class="fa-solid fa-play"></i> Resume`;
            btnPauseCamera.className = "";
            cameraCardContainer.classList.remove('scanning');
            statWebcam.textContent = "Paused";
            statWebcam.style.color = "#f59e0b";
            updateSystemStatus('loading', 'Camera Paused');
            
            // Clear overlay canvas
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
    });

    // HTML escape helper
    function escapeHtml(unsafe) {
        return unsafe
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#039;");
    }

    // Run Initializer
    init();
});
