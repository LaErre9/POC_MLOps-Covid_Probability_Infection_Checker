from flask import Flask, render_template, request
app = Flask(__name__)
import pickle


file = open('model.pkl', 'rb')

clf = pickle.load(file)
file.close()
@app.route('/', methods=["GET", "POST"])
def covid_checker():
    if request.method == "POST":
        myDict = request.form
        breating = int(myDict['Breathing Problem'])
        fever = int(myDict['Fever'])
        dry = int(myDict['Dry Cough'])
        sore = int(myDict['Sore throat'])
        running = int(myDict['Running Nose'])
        asthma = int(myDict['Asthma'])
        chronic = int(myDict['Chronic Lung Disease'])
        headache = int(myDict['Headache'])
        heart = int(myDict['Heart Disease'])
        diabetes = int(myDict['Diabetes'])
        hyper = int(myDict['Hyper Tension'])
        fatigue = int(myDict['Fatigue '])
        gastrointestinal = int(myDict['Gastrointestinal '])
        abroad = int(myDict['Abroad travel'])
        contact = int(myDict['Contact with COVID Patient'])
        attended = int(myDict['Attended Large Gathering'])
        visited = int(myDict['Visited Public Exposed Places'])
        family = int(myDict['Family working in Public Exposed Places'])
        wearing = int(myDict['Wearing Masks'])
        sanitization = int(myDict['Sanitization from Market'])
        # Code for Inference
        inputFeatures = [breating, fever, dry, sore, running, asthma, chronic, headache,
                heart, diabetes, hyper, fatigue, gastrointestinal, abroad, contact, 
                attended, visited, family, wearing, sanitization]
        infProb =clf.predict_proba([inputFeatures])[0][1]
        #print(infProb)
        if infProb >= 0 and infProb <= 0.50:
            str1 = " Pertanto la Sua situazione non desta preoccupazione verso l'infezione al SARS-CoV-2 (COVID-19). \
            Se comunque la preoccupazione sussiste è bene chiamare il proprio medico di fiducia/famiglia per informazioni \
                più dettagliate." 
            return render_template('show.html', inf = round((infProb*100), 0), text = str1)
        elif infProb > 0.50 and infProb <= 0.75:
            str2 = " Pertanto la Sua situazione è dubbia, riprovi a fare il test oppure chiami il Suo medico di famiglia, \
                il Suo pediatra o la guardia medica per avere informazioni più dettagliate."
            return render_template('show.html', inf = round((infProb*100), 0), text = str2)
        elif infProb > 0.75 and infProb <= 1:
            str3 = " Pertanto la Sua situazione suscita preoccupazione e per il test è stato infettato dal SARS-CoV-2 (COVID-19). \
                Tuttavia, rimanga in casa, non si rechi al pronto soccorso o presso gli studi medici, ma chiami al telefono il Suo medico di famiglia, \
                il Suo pediatra o la guardia medica. Oppure chiami il Numero Verde regionale oppure ancora al Numero di Pubblica utilità: 1500."
            return render_template('show.html', inf = round((infProb*100), 0), text = str3) 
    return render_template('index.html')
    
   # return 'Hello, World!' + str(infProb)


if __name__=="__main__":
    app.run(debug=True)