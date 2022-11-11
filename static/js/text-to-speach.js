const synth = window.speechSynthesis;
var voices = []
const msg = new SpeechSynthesisUtterance();
synth.cancel();

/*
/* SOLO PER CHROME
/*
/* Per abilitare lo speach appena carica la pagina andare nella impostazioni chrome in
/* chrome://settings/content/sound e aggiungere la pagine nell'ultima sezione "Possono riprodurre suoni"
*/
function onLoad(){
    if (synth.paused === true) {
        synth.resume();
        return;
    }
    const div = document.getElementById("dialogo");
    textToSpeech(div.innerText);
    synth.cancel;
}

function textToSpeech(text) {
    voices = synth.getVoices();
    msg.text = text;
    msg.voice = voices[11];
    msg.rate = 1.2;

    synth.speak(msg);
}