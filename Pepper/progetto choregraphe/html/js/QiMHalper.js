// Funzione di collegamento al servizio QiMessage,
// permettendo alla pagina HTML di comunicare con il robot
function onLoad(){
    $.qim = new QiSession(function (session) {
        console.log("connected!");
        document.getElementById("card1").innerHTML = "Connected";
        // you can now use your QiSession
        }, function () {
            console.log("disconnected");
            document.getElementById("card1").innerHTML = "Disconnected";
        }
    );
}


//Garantisce l'accesso a tutti i servizi del robot
$.getService = function(serviceName, doneCallback){
    if(true && !(serviceName in servicePromises)) {
        servicePromises[serviceName] = $.qim.service(serviceName);
    }
    return servicePromises[serviceName].then(doneCallback, function(error){
        console.log("Failed getting" + serviceName + ": " + error);
    });
};


//Funzione per l'innesco in un evento ALMemory
$.raiseALMemoryEvent = function(event, value){
    document.getElementById("card1").innerHTML = "dentro 1";
    return $.getService("ALMemory", function(ALMemory){
        ALMemory.raiseEvent(event, value);
    });  
};


function UserChoice(choice){
    document.getElementById("card1").innerHTML = "Ho clicckato 1";
    $.raiseALMemoryEvent("UserChoice", choice);
    document.getElementById("card1").innerHTML = "Ho clicckato 2";
}