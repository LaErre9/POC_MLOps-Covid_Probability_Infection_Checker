<div align = "center" style = "float:left; margin - right:1em;" ><img src="https://raw.githubusercontent.com/LaErre9/POC_MLOps-Covid_Probability_Infection_Checker/main/templates/covid_probability_detector.png?token=AB475L7VVLSSF7KDB3NM25DBIODCE" alt = "covid-icon-probabilty" width="130" height="auto" style="margin-top: 15px; margin-bottom: -20px; float: left;"> <img src="https://raw.githubusercontent.com/LaErre9/POC_MLOps-Covid_Probability_Infection_Checker/main/templates/LogoMLOPs_ARinvented.png?token=AB475L25BUZU7OU25EXPE6TBIXAIW" alt = "MLOps-icon-AR" width="auto" height="130" style="margin-top: 15px; margin-bottom: -20px; margin-left: 20px;"></div>
     
# Proof of Concept - MLOps / COVID probability infection checker
Il seguente repository è una semplice realizzazione di bozza progettuale per tracciare un progetto supportato dalle pratiche **MLOps**, utile per testare l’idea o l’ipotesi di progetto al fine di dimostrarne la fattibilità.

## L'idea
L’ idea è quella di predire una certa probabilità di infezione al COVID-19 in base ai sintomi o alle malattie che il paziente manifesta e guidarlo a come comportarsi in caso di alta probabilità di infezione. Ad esempio, se il sistema predice che il paziente X abbia una probabilità di infezione tra il 75% e il 100% allora verrà consigliato di consultare immediatamente un operatore sanitario e di rimanere nel suo domicilio. Se il sistema predice che il paziente Y abbia una probabilità di infezione tra il 50% e il 75% allora verrà consigliato di auto medicarsi in casa e se vuole può chiamare il medico di fiducia per ulteriori consultazioni utili per sollevarsi dai sintomi. Il modello ML sarà inglobato in una Web App in modo tale che sarà facilmente usabile dagli utenti. 

## La Tesi
Si è nel 2021, e si è ormai capito come l’Intelligenza Artificiale si sia evoluta, e si sta ancora evolvendo! dimostrandosi di notevole importanza in tantissimi settori. Secondo alcune fonti, il 60% delle medie grandi organizzazioni Italiane ha attivato almeno un progetto software che implementa l’AI.
Progetto software che deve essere opportunamente supportato, realizzato al fine di garantire benefici ed efficacia per le organizzazioni stesse. Pertanto, l’ingegneria del software ha la necessità di scoprire una metodologia o un insieme di pratiche utili per la realizzazione e per il supporto di prodotti software che all’interno di esso implementino un modello di apprendimento dell’IA: il **Machine Learning**. Dunque è questa una delle tante motivazioni che mi hanno spinto ad approfondire ad una nuovissima disciplina di progettazione del software, il MLOps. Nonché il titolo della mia Tesi dove ho raccolto principi, caratteristiche e sperimentato un’applicazione.

Il MLOps è un’estensione di DevOps, oserei dire la reincarnazione di DevOps, perchè nasce con l’obiettivo di colmare le carenze legate all’interdipendenza del software tradizionale (e quindi del classico ciclo di sviluppo del software) con l’integrazione, l’automazione e il monitoraggio in tutte le fasi della costruzione del modello ML con l’obiettivo di farlo funzionare continuamente in produzione, includendo tutte le fasi delle best practices, test automatico, rilascio automatico e distribuzione del software. MLOps ha tantissime sfaccettature o Vantaggi: una su tutte è il coinvolgimento di diversi team di un’organizzazione, il team di sviluppo aziendale, il team degli ingegneri dei dati, data scientist, team degli ingegneri DevOps. Importante è anche la suddivisione dei livelli di implementazione del MLOps sulla base di come si evolve nel tempo il modello ML. Più il modello ML opererà in ambienti in continua evoluzione, più il livello di implementazione sale. 

Esistono tantissime applicazioni MLOps. 
Ho voluto applicare i principi MLOps in una possibile applicazione, e credo sia anche un caso di notevolissima importanza, soprattutto in questo periodo di pandemia mondiale, ovvero in ambito Healtcare (sanità). Grazie alla quantità di dati medici generati oggigiorno, le organizzazioni sanitarie possono sfruttare questa mole di conoscenza per fornire risultati migliori con maggiore precisione. Infatti, la realizzazione di software che ingloba ML che tratta questi dati, può aiutare le organizzazioni sanitare a fornire le previsioni e i risultati più semplici per i pazienti, riducendo i costi e personalizzando in modo significativo l’assistenza ai pazienti. 

Pertanto, ho sviluppato e trattato un *Proof of Concept* delle best practices MLOps in un semplice applicativo (una Web App) in grado di predire se una persona è stata infetta dal virus Covid-19 sulla base dei sintomi, malattie e comportamenti e dare dei consigli su come comportarsi in base al risultato ottenuto. Ho seguito quindi i principi dell MLOps, sperimentandone l’implementazione sia dal lato della implementazione del modello ML dove ho usato tools come GitHub (per il repository e versionamento del codice), GitHub Actions (Automazione della pipeline ML: CI, CDel, CD), DVC (Versionamento dei dati e del modello ML), CML (Continuous Machine Learning per il continuo monitoraggio) e sia dal lato dell’implementazione della Web App dove ho realizzato un form dal quale l’utente inserirà i dati, che verranno prelevati da un file python (ho lavorato con il framework Flask per permettere tutto questo), che a sua volta li passerà in un file python opportunamente programmato per il calcolo della predizione attraverso un opportuno algoritmo di Machine Learning. Infine ho utilizzato Heroku per la distribuzione dell’applicazione in maniera automatica, ovvero, ad ogni modifica o aggiornamento che verrà fatto sul main branch di questo repository GitHub, verrà distribuita in automatico la versione aggiornata dell’applicazione. Applicazione che sarà disponibile online e usufruibile dagli utenti. 

In definitiva, questi principi risultano essere necessari per lo sviluppo di software basato su algoritmi ML al fine di rendere efficiente e veloce la produzione di tali sistemi software all’interno delle organizzazioni che implementeranno sistemi che si adattano all’ambiente in maniera automatica, di sistemi che migliorano le proprie prestazioni rispetto allo svolgimento di un particolare compito oppure sistemi che scoprono modelli o nuove informazioni a partire da dati empirici. 


## La Web App
La Wev App è disponibile al seguente link: https://covid-probability-checker.herokuapp.com/ 
oppure nella *home* del repository e nella sezione *Environments* cliccare **covid-probability-checker** > **View deployment** <br>

**Screen della Web App:**

<p align = "center">
<img src="https://raw.githubusercontent.com/LaErre9/POC_MLOps-Covid_Probability_Infection_Checker/main/templates/screenWebApp.png?token=AB475LYA3AE6GEN74PCL5FDBIXESO" alt = "covid-icon-probabilty" width="auto" height="auto" align="middle" style="margin-top: 15px; margin-bottom: -20px;">
</p><br>

:warning: **ATTENZIONE** *L’applicazione realizzata dall’autore è solo a scopo dimostrativo, pertanto la probabilità calcolata può risultare incerta giacché, come già detto in precedenza, il dataset è datato e non aggiornato e i sintomi del virus mutano e variano ogni giorno*

**Il funzionamento:** <br>

<p align = "center">
<img src="https://raw.githubusercontent.com/LaErre9/POC_MLOps-Covid_Probability_Infection_Checker/main/templates/scheme_function.png?token=AB475L2KW3GNVL26SLBOT4DBI5F3O" alt = "covid-icon-probabilty" width="auto" height="auto" align="middle" style="margin-top: 15px; margin-bottom: -20px;">
</p>

## Il progetto
Il progetto è stato sviluppato seguendo i principi dell' *MLOps* utilizzando i seguenti tools:
- Pipeline DVC
- CI/CD con GitHub Actions
- CML con GitHub Actions
- Distribuzione e Rilascio con GitHub Actions e Heroku (distribuzione Web App) <br>

*Il flusso del POC MLOps è*: <br>

<p align = "center">
<img src="https://raw.githubusercontent.com/LaErre9/POC_MLOps-Covid_Probability_Infection_Checker/main/templates/flussoPOCMLops.png?token=AB475L62GO53TX4WEPZOTNTBIXCEK" alt = "covid-icon-probabilty" width="auto" height="auto" align="middle" style="margin-top: 15px; margin-bottom: -20px;">
</p>


## Progetto realizzato per soli scopi dimostrativi ed educativi
Il seguente repository è stato realizzato come supporto alla tesi di Laurea Triennale: **Machine Learning Operations: principi, caratteristiche ed applicazione**, *Università degli Studi di Napoli, Federico II*, realizzato da <a title="Antonio Romano" href="https://github.com/LaErre9" target="_blank" >Antonio Romano</a>

