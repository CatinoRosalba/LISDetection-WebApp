const synth = window.speechSynthesis;
const voices = synth.getVoices();
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
    const msg = new SpeechSynthesisUtterance();
    msg.text = text;
    msg.voice = voices.filter(function (voice) { return voice.name === "Google italiano"; })[0];

    synth.speak(msg);
}