import spacy
import pickle


nlp = spacy.load('models/try20/model-best')
nlp.add_pipe('sentencizer')
text = r"Sigríður Ólafsdóttir kemur frá landi sem heitir Noregur. Á æskuárunum naut hann þess að gera ýmislegt í heimabæ sínum Ósló. Hann nýtur tilfinningarinnar um 100 dollara seðla í hendi sér á fimmtudagseftirmiðdegi. Hann fæddist 4. júlí 1996 og árið 2016 hóf hann BA gráðu við háskólann í Bergen. Hann útskrifaðist 01.07.2019 og 01.09.2020 hóf hann meistaranám í París, Frakklandi. hann á einn bróður, Haraldur Benediktsson, sem nýtur líka 50 punda seðla í höndunum. Imran á nokkur hlutabréf, hann á 50% í Google, Kahoot og Ólympíuleikunum"
doc = nlp(text)
for sent in doc.sents:
    print(sent, sent.ents)
for ent in doc.ents:
    print(ent.text, ent.label_)
