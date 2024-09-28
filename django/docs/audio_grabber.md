disable_toc: true

<h1>Audio Grabber</h1>
<div class="form-group">
    <label for="tenant_id" class="control-label">Tenant ID</label>
    <input type="text" id="tenant_id" class="form-control" placeholder="Tenant ID" value="0000">
</div>

<div class="form-group">
    <label for="audioInputSelect" class="control-label">Audio Input</label>
    <select id="audioInputSelect" class="form-control"></select>
</div>

<div class="form-group">
    <label for="translate_from" class="control-label">Translate From</label>
    <select id="translate_from" class="form-control">
    <option value="af" >Afrikaans</option>
    <option value="sq" >Albanian</option>
    <option value="am" >Amharic</option>
    <option value="ar" >Arabic</option>
    <option value="hy" >Armenian</option>
    <option value="ast" >Asturian</option>
    <option value="az" >Azerbaijani</option>
    <option value="ba" >Bashkir</option>
    <option value="be" >Belarusian</option>
    <option value="bn" >Bengali</option>
    <option value="bs" >Bosnian</option>
    <option value="br" >Breton</option>
    <option value="bg" >Bulgarian</option>
    <option value="my" >Burmese</option>
    <option value="ca" >Catalan</option>
    <option value="ceb" >Cebuano</option>
    <option value="km" >Central Khmer</option>
    <option value="zh" >Chinese</option>
    <option value="hr" >Croatian</option>
    <option value="cs" >Czech</option>
    <option value="da" >Danish</option>
    <option value="nl" >Dutch</option>
    <option value="en" selected>English</option>
    <option value="et" >Estonian</option>
    <option value="fi" >Finnish</option>
    <option value="fr" >French</option>
    <option value="ff" >Fulah</option>
    <option value="gd" >Gaelic</option>
    <option value="gl" >Galician</option>
    <option value="lg" >Ganda</option>
    <option value="ka" >Georgian</option>
    <option value="de">German</option>
    <option value="el" >Greeek</option>
    <option value="gu" >Gujarati</option>
    <option value="ht" >Haitian</option>
    <option value="ha" >Hausa</option>
    <option value="he" >Hebrew</option>
    <option value="hi" >Hindi</option>
    <option value="hu" >Hungarian</option>
    <option value="is" >Icelandic</option>
    <option value="ig" >Igbo</option>
    <option value="ilo" >Iloko</option>
    <option value="id" >Indonesian</option>
    <option value="ga" >Irish</option>
    <option value="it" >Italian</option>
    <option value="ja" >Japanese</option>
    <option value="jv" >Javanese</option>
    <option value="kn" >Kannada</option>
    <option value="kk" >Kazakh</option>
    <option value="ko" >Korean</option>
    <option value="lo" >Lao</option>
    <option value="lv" >Latvian</option>
    <option value="ln" >Lingala</option>
    <option value="lt" >Lithuanian</option>
    <option value="lb" >Luxembourgish</option>
    <option value="mk" >Macedonian</option>
    <option value="mg" >Malagasy</option>
    <option value="ms" >Malay</option>
    <option value="ml" >Malayalam</option>
    <option value="mr" >Marathi</option>
    <option value="mn" >Mongolian</option>
    <option value="ne" >Nepali</option>
    <option value="ns" >Northern Sotho</option>
    <option value="no" >Norwegian</option>
    <option value="oc" >Occitan (post 1500)</option>
    <option value="or" >Oriya</option>
    <option value="pa" >Panjabi</option>
    <option value="fa" >Persian</option>
    <option value="pl" >Polish</option>
    <option value="pt" >Portuguese</option>
    <option value="ps" >Pushto</option>
    <option value="ro" >Romanian</option>
    <option value="ru" >Russian</option>
    <option value="sr" >Serbian</option>
    <option value="sd" >Sindhi</option>
    <option value="si" >Sinhala</option>
    <option value="sk" >Slovak</option>
    <option value="sl" >Slovenian</option>
    <option value="so" >Somali</option>
    <option value="es" >Spanish</option>
    <option value="su" >Sundanese</option>
    <option value="sw" >Swahili</option>
    <option value="ss" >Swati</option>
    <option value="sv" >Swedish</option>
    <option value="tl" >Tagalog</option>
    <option value="ta" >Tamil</option>
    <option value="th" >Thai</option>
    <option value="tn" >Tswana</option>
    <option value="tr" >Turkish</option>
    <option value="uk" >Ukrainian</option>
    <option value="ur" >Urdu</option>
    <option value="uz" >Uzbek</option>
    <option value="vi" >Vietnamese</option>
    <option value="cy" >Welsh</option>
    <option value="fy" >Western Frisian</option>
    <option value="wo" >Wolof</option>
    <option value="xh" >Xhosa</option>
    <option value="yi" >Yiddish</option>
    <option value="yo" >Yoruba</option>
    <option value="zu" >Zulu</option>
    </select>
</div>

<div class="form-group">
    <label for="translate_to" class="control-label">Translate To</label>
    <select id="translate_to" class="form-control">
    <option value="_" selected>&#60;no translation&#62;</option>
    <option value="af" >Afrikaans</option>
    <option value="sq" >Albanian</option>
    <option value="am" >Amharic</option>
    <option value="ar" >Arabic</option>
    <option value="hy" >Armenian</option>
    <option value="ast" >Asturian</option>
    <option value="az" >Azerbaijani</option>
    <option value="ba" >Bashkir</option>
    <option value="be" >Belarusian</option>
    <option value="bn" >Bengali</option>
    <option value="bs" >Bosnian</option>
    <option value="br" >Breton</option>
    <option value="bg" >Bulgarian</option>
    <option value="my" >Burmese</option>
    <option value="ca" >Catalan</option>
    <option value="ceb" >Cebuano</option>
    <option value="km" >Central Khmer</option>
    <option value="zh" >Chinese</option>
    <option value="hr" >Croatian</option>
    <option value="cs" >Czech</option>
    <option value="da" >Danish</option>
    <option value="nl" >Dutch</option>
    <option value="en" >English</option>
    <option value="et" >Estonian</option>
    <option value="fi" >Finnish</option>
    <option value="fr" >French</option>
    <option value="ff" >Fulah</option>
    <option value="gd" >Gaelic</option>
    <option value="gl" >Galician</option>
    <option value="lg" >Ganda</option>
    <option value="ka" >Georgian</option>
    <option value="de">German</option>
    <option value="el" >Greeek</option>
    <option value="gu" >Gujarati</option>
    <option value="ht" >Haitian</option>
    <option value="ha" >Hausa</option>
    <option value="he" >Hebrew</option>
    <option value="hi" >Hindi</option>
    <option value="hu" >Hungarian</option>
    <option value="is" >Icelandic</option>
    <option value="ig" >Igbo</option>
    <option value="ilo" >Iloko</option>
    <option value="id" >Indonesian</option>
    <option value="ga" >Irish</option>
    <option value="it" >Italian</option>
    <option value="ja" >Japanese</option>
    <option value="jv" >Javanese</option>
    <option value="kn" >Kannada</option>
    <option value="kk" >Kazakh</option>
    <option value="ko" >Korean</option>
    <option value="lo" >Lao</option>
    <option value="lv" >Latvian</option>
    <option value="ln" >Lingala</option>
    <option value="lt" >Lithuanian</option>
    <option value="lb" >Luxembourgish</option>
    <option value="mk" >Macedonian</option>
    <option value="mg" >Malagasy</option>
    <option value="ms" >Malay</option>
    <option value="ml" >Malayalam</option>
    <option value="mr" >Marathi</option>
    <option value="mn" >Mongolian</option>
    <option value="ne" >Nepali</option>
    <option value="ns" >Northern Sotho</option>
    <option value="no" >Norwegian</option>
    <option value="oc" >Occitan (post 1500)</option>
    <option value="or" >Oriya</option>
    <option value="pa" >Panjabi</option>
    <option value="fa" >Persian</option>
    <option value="pl" >Polish</option>
    <option value="pt" >Portuguese</option>
    <option value="ps" >Pushto</option>
    <option value="ro" >Romanian</option>
    <option value="ru" >Russian</option>
    <option value="sr" >Serbian</option>
    <option value="sd" >Sindhi</option>
    <option value="si" >Sinhala</option>
    <option value="sk" >Slovak</option>
    <option value="sl" >Slovenian</option>
    <option value="so" >Somali</option>
    <option value="es" >Spanish</option>
    <option value="su" >Sundanese</option>
    <option value="sw" >Swahili</option>
    <option value="ss" >Swati</option>
    <option value="sv" >Swedish</option>
    <option value="tl" >Tagalog</option>
    <option value="ta" >Tamil</option>
    <option value="th" >Thai</option>
    <option value="tn" >Tswana</option>
    <option value="tr" >Turkish</option>
    <option value="uk" >Ukrainian</option>
    <option value="ur" >Urdu</option>
    <option value="uz" >Uzbek</option>
    <option value="vi" >Vietnamese</option>
    <option value="cy" >Welsh</option>
    <option value="fy" >Western Frisian</option>
    <option value="wo" >Wolof</option>
    <option value="xh" >Xhosa</option>
    <option value="yi" >Yiddish</option>
    <option value="yo" >Yoruba</option>
    <option value="zu" >Zulu</option>
    </select>
</div>

<div class="form-group">
    <button id="startBtn" class="btn btn-success">Start</button>
    <button id="stopBtn" class="btn btn-danger" disabled>Stop</button>
</div>

<div class="form-group">
    <canvas id="spectrogram" class="form-control" style="width: 100%; height: 50px;"></canvas>
</div>
<div class="form-group">
    <canvas id="volumeMeter" class="form-control" style="width: 100%; height: 50px;"></canvas>
</div>

<div id="recording-message" class="alert alert-info" style="display: none;">
    Leave this window open to continue the recording.
</div>

<div id="popup-message" class="alert alert-warning" style="display: none;">
    Click <button id="popup-link" class="btn btn-primary">this link</button> to open a pop-up window that contains the transcript/translation.
</div>


<script>
    let audioContext;
    let mediaStream;
    let scriptProcessor;
    let buffer = [];
    let chunkId = Date.now().toString();
    let recording = false;
    let analyser;

    const RATE = 16000;
    const BUFFER_SIZE = 2 * 10 * RATE; // 10 seconds of audio
    const SILENCE_THRESHOLD = 300;

    const spectrogramCanvas = document.getElementById('spectrogram');
    const volumeMeterCanvas = document.getElementById('volumeMeter');
    const spectrogramCtx = spectrogramCanvas.getContext('2d');
    const volumeMeterCtx = volumeMeterCanvas.getContext('2d');

    document.getElementById('startBtn').addEventListener('click', startRecording);
    document.getElementById('stopBtn').addEventListener('click', stopRecording);

    // Get the list of audio input devices and populate the select element
    navigator.mediaDevices.enumerateDevices().then(devices => {
        const audioInputSelect = document.getElementById('audioInputSelect');
        const desktopOption = document.createElement('option');
        desktopOption.value = 'desktop';
        desktopOption.text = 'Desktop Audio';
        audioInputSelect.appendChild(desktopOption);

        devices.forEach(device => {
            if (device.kind === 'audioinput') {
                const option = document.createElement('option');
                option.value = device.deviceId;
                option.text = device.label || `Microphone ${audioInputSelect.length + 1}`;
                audioInputSelect.appendChild(option);
            }
        });
    });

    async function startRecording() {
        const selectedDeviceId = document.getElementById('audioInputSelect').value;
        const tenantId = document.getElementById('tenant_id').value;

        // Show the messages when recording starts
        document.getElementById('recording-message').style.display = 'block';
        document.getElementById('popup-message').style.display = 'block';

        // Add event listener to open the pop-up window        
        document.getElementById('popup-link').addEventListener('click', function(event) {
            event.preventDefault();
            const popupUrl = `/translator_susi_ai_iframe.html?tenant_id=${tenantId}`;
            window.open(popupUrl, 'TranscriptWindow', 'width=600,height=400');
        });

        if (selectedDeviceId === 'desktop') {
            try {
                const stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
                const audioTrack = stream.getAudioTracks()[0];
                const audioOnlyStream = new MediaStream([audioTrack]);
                startStream(audioOnlyStream);
            } catch (error) {
                console.error('Error accessing desktop audio', error);
            }
        } else {
            const constraints = {
                audio: {
                    deviceId: selectedDeviceId ? { exact: selectedDeviceId } : undefined,
                    sampleRate: RATE,
                },
            };

            navigator.mediaDevices.getUserMedia(constraints)
                .then(stream => {
                    startStream(stream);
                })
                .catch(error => console.error('Error accessing audio device', error));
        }
    }

    function startStream(stream) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: RATE });
        mediaStream = stream;
        const mediaStreamSource = audioContext.createMediaStreamSource(stream);
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;
        //scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
        scriptProcessor = audioContext.createScriptProcessor(8192, 1, 1);

        mediaStreamSource.connect(analyser);
        mediaStreamSource.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);
        scriptProcessor.onaudioprocess = processAudio;

        recording = true;
        document.getElementById('startBtn').disabled = true;
        document.getElementById('stopBtn').disabled = false;

        drawSpectrogram();
        drawVolumeMeter();
    }

    function stopRecording() {
        recording = false;
        mediaStream.getTracks().forEach(track => track.stop());
        scriptProcessor.disconnect();
        document.getElementById('startBtn').disabled = false;
        document.getElementById('stopBtn').disabled = true;
    }

    function processAudio(event) {
        const audioData = event.inputBuffer.getChannelData(0);
        if (isSilent(audioData)) {
            buffer = []; // Reset buffer
            chunkId = Date.now().toString(); // Get new chunk ID
        } else {
            buffer.push(...audioData);
        }

        if (buffer.length > 0) {
            sendChunk();
        }

        if (buffer.length >= BUFFER_SIZE) {
            buffer = []; // Reset buffer
            chunkId = Date.now().toString(); // Get new chunk ID
        }
    }

    function isSilent(data) {
        const maxVal = Math.max(...data);
        return maxVal < SILENCE_THRESHOLD / 32767; // Convert to 16-bit equivalent threshold
    }

    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }

    function sendChunk() {
        const int16Array = new Int16Array(buffer.map(n => n * 32767));
        const audioBuffer = new Blob([int16Array.buffer], { type: 'audio/wav' });
        const reader = new FileReader();
        reader.readAsDataURL(audioBuffer);
        reader.onloadend = () => {
            const base64data = reader.result.split(',')[1];
            const data = {
                tenant_id: document.getElementById('tenant_id').value,
                translate_from: document.getElementById('translate_from').value,
                translate_to: document.getElementById('translate_to').value,
                chunk_id: chunkId,
                audio_b64: base64data
            };

            // construct the URL from the host and port
            const transcribeurl = `/api/transcribe`;
            const csrftoken = getCookie('csrftoken');
            fetch(transcribeurl, {
                method: 'POST',
                headers: { 'X-CSRFToken': csrftoken, 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            })
                .then(response => {
                    if (response.ok) {
                        console.log(`Sent chunk ${chunkId} with ${buffer.length} samples`);
                    } else {
                        console.error(`Error sending chunk: ${response.status}:${response.statusText}`);
                    }
                })
                .catch(error => console.error('Error sending chunk:', error));
        };
    }

    function getColor(value) {
        const percent = value / 255;
        const red = Math.floor(Math.max(0, 255 * (percent - 0.5) * 2));
        const green = Math.floor(Math.max(0, 255 * (0.5 - Math.abs(percent - 0.5)) * 2));
        const blue = Math.floor(Math.max(0, 255 * (0.5 - percent) * 2));
        return `rgb(${red}, ${green}, ${blue})`;
    }

    function drawSpectrogram() {
        if (!recording) return;

        const freqData = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(freqData);

        const width = spectrogramCanvas.width;
        const height = spectrogramCanvas.height;

        // Shift existing image to the left
        const imageData = spectrogramCtx.getImageData(1, 0, width - 1, height);
        spectrogramCtx.putImageData(imageData, 0, 0);

        // Draw new frequency data on the right
        for (let i = 0; i < height; i++) {
            const value = freqData[i];
            spectrogramCtx.fillStyle = getColor(value);
            spectrogramCtx.fillRect(width - 1, height - i, 1, 1);
        }

        requestAnimationFrame(drawSpectrogram);
    }

    function drawVolumeMeter() {
        if (!recording) return;

        const timeData = new Uint8Array(analyser.fftSize);
        analyser.getByteTimeDomainData(timeData);

        const width = volumeMeterCanvas.width;
        const height = volumeMeterCanvas.height;

        // Shift existing image to the left
        const imageData = volumeMeterCtx.getImageData(1, 0, width - 1, height);
        volumeMeterCtx.putImageData(imageData, 0, 0);

        // Calculate volume
        const volume = Math.sqrt(timeData.reduce((sum, value) => sum + Math.pow(value - 128, 2), 0) / timeData.length);
        const volumeHeight = (volume / 32) * height;

        // Clear the volume on the right with blue color
        volumeMeterCtx.fillStyle = 'grey';
        volumeMeterCtx.fillRect(width - 1, 0, 1, height);

        // Draw new volume level on the right
        volumeMeterCtx.fillStyle = 'black';
        volumeMeterCtx.fillRect(width - 1, height - volumeHeight, 1, volumeHeight);

        requestAnimationFrame(drawVolumeMeter);
    }

    function adjustCanvasSize() {
        const spectrogramCanvas = document.getElementById('spectrogram');
        const volumeMeterCanvas = document.getElementById('volumeMeter');
    
        const containerWidth = document.querySelector('.form-group').offsetWidth;
    
        spectrogramCanvas.width = containerWidth;
        volumeMeterCanvas.width = containerWidth;
    }

    window.addEventListener('resize', adjustCanvasSize);
    window.addEventListener('load', adjustCanvasSize);

</script>