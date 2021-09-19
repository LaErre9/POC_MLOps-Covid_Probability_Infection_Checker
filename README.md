<div align = "center" style = "float:left; margin - right:1em;" ><img src="https://raw.githubusercontent.com/LaErre9/POC_MLOps-Covid_Probability_Infection_Checker/main/templates/covid_probability_detector.png?token=AB475L7VVLSSF7KDB3NM25DBIODCE" alt = "covid-icon-probabilty" width="130" height="auto" style="margin-top: 15px; margin-bottom: -20px; float: left;"> <img src="https://raw.githubusercontent.com/LaErre9/POC_MLOps-Covid_Probability_Infection_Checker/main/templates/LogoMLOPs_ARinvented.png?token=AB475L25BUZU7OU25EXPE6TBIXAIW" alt = "MLOps-icon-AR" width="auto" height="130" style="margin-top: 15px; margin-bottom: -20px; margin-left: 20px;"></div>
     
# Proof of Concept - MLOps / COVID probability infection checker
Il seguente repository è una semplice realizzazione di bozza progettuale per tracciare un progetto supportato dalle pratiche **MLOps**, utile per testare l’idea o l’ipotesi di progetto al fine di dimostrarne la fattibilità.

## L'idea
L’ idea è quella di predire una certa probabilità di infezione al COVID-19 in base ai sintomi o alle malattie che il paziente manifesta e guidarlo a come comportarsi in caso di alta probabilità di infezione. Ad esempio, se il sistema predice che il paziente X abbia una probabilità di infezione tra il 75% e il 100% allora verrà consigliato di consultare immediatamente un operatore sanitario e di rimanere nel suo domicilio. Se il sistema predice che il paziente Y abbia una probabilità di infezione tra il 50% e il 75% allora verrà consigliato di auto medicarsi in casa e se vuole può chiamare il medico di fiducia per ulteriori consultazioni utili per sollevarsi dai sintomi. Il modello ML sarà inglobato in una Web App in modo tale che sarà facilmente usabile dagli utenti. 

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
Il seguente repository è stato realizzato come supporto alla tesi di Laurea: *Machine Learning Operations: principi, caratteristiche ed applicazione*, Università degli Studi di Napoli, Federico II,  realizzato da <a title="Antonio Romano" href="https://github.com/LaErre9" target="_blank" >Antonio Romano</a>

