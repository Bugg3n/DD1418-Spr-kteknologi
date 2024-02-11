from gettext import npgettext
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import os
from Reader import Reader
import argparse
import numpy as np
from ViterbiTrigramDecoder import ViterbiTrigramDecoder

# The mapping from swedish to english.
translation = {}

def scrape_word(word):
    s = Service("/usr/lib/chromium-browser/chromedriver")
    driver = webdriver.Chrome(service=s)
    driver.get("https://sv.bab.la/lexikon/svensk-engelsk/" + word)

    translation_list = []

    # Extracting data
    first_word = ""
    content = driver.page_source
    number_of_translations = 0
    soup = BeautifulSoup(content, features="html.parser")
    for a in soup.find_all('div', href=False, attrs={'class':'quick-result-entry'}):
            for b in a.find_all('a', href=True, attrs={'class':'babQuickResult'}):
                if b.text == word or first_word:
                    for c in a.find_all('a',href=True, attrs={'class':'scroll-link'}, onclick=False):                        
                        translation_list.append(c.text.lower())
                        number_of_translations += 1
                if number_of_translations == 0:
                    first_word = b.text
                    for c in a.find_all('a',href=True, attrs={'class':'scroll-link'}, onclick=False):
                        translation_list.append(c.text.lower())
                        number_of_translations += 1
                    
    for i in translation_list:
        if i == "arrow_upward":
            translation_list.remove(i)
    translation_list.append(word)
    return translation_list

def scrape_inflections(en_word, sw_word):
    s = Service("/usr/lib/chromium-browser/chromedriver")
    driver = webdriver.Chrome(service=s)
    driver.get("https://www.britannica.com/dictionary/" + en_word)
    inflection_list = []
    # Extracting data
    content = driver.page_source
    soup = BeautifulSoup(content, features="html.parser")
    for a in soup.findAll('span', href=False, attrs={'class':'i_text'}):
        inflection = remove_punctuation(a.text).lower()
        if inflection not in inflection_list and inflection not in translation[sw_word]:
            inflection_list.append(inflection)
    return inflection_list

def remove_punctuation(word):
    newWord = []
    word_list = [*word]
    for i in word_list:
        if i.isalpha():
            newWord.append(i)
    return "".join(newWord)      

def add_inflections(sw_word):
    inflections = []
    for en_word in translation[sw_word][0:-1]:
        inflection_list = scrape_inflections(en_word, sw_word)
        for j in inflection_list:
            inflections.append(j)
    for k in inflections:
        translation[sw_word].append(k)

def get_translation(word, filename_t):
    # Check if translation in file |Swedish word,translation1, translation2,...     
    if word in translation.keys():
        return translation[word]
    else:
        translation[word] = scrape_word(word)
        add_inflections(word)
        with open(filename_t, 'a', encoding='utf-8') as file:
            file.write(word + ",")
            for i in translation[word]:
                file.write(i + ",")
            file.write("\n")
        return (translation[word])

def read_file(filename):
    try:
        f = open(filename, 'r', encoding='utf-8')
        for line in f.readlines():
            word_list = line.split(",")
            translation[word_list[0]] = word_list[1:len(word_list)-1]
    except IOError:
        # Creates a new file
        path = os.getcwd()
        os.listdir(path) 
        with open(filename, 'w', encoding='utf-8') as file:
            pass
        os.listdir(path)

def n_gram_translate(bigram_probs, trigram_probs, to_translate_list, direct_translation_list):
    if len(to_translate_list) == 1:  # If one word, direct ]translate
        return [translation[to_translate_list[0]][0]]
    elif len(to_translate_list) == 2:  # If two words, use bigram probabilities
        return bigram_translate(bigram_probs, to_translate_list)
    elif len(to_translate_list) < 6:  # If more than that, use trigram probabilites
        return trigram_translate(trigram_probs, to_translate_list, direct_translation_list, bigram_probs)
    elif len(to_translate_list) >= 6:
        i = 0
        returnlist = []
        while i < len(to_translate_list)-4:
            returnlist+=(trigram_translate(trigram_probs, to_translate_list[i:i+5], direct_translation_list[i:i+5], bigram_probs))
            i+=5
        if i != len(to_translate_list):
            returnlist+=(n_gram_translate(trigram_probs, to_translate_list[i:-1], direct_translation_list[i:-1], bigram_probs))
        return returnlist
    else:
        return[]

def trigram_translate(trigram_probs, to_translate_list, direct_translation_list, bigram_probs):
    best_sentance_list = direct_translation_list
    candidate_array = []
    for word in to_translate_list:
        translation_candidates = []
        try:
            for translation_candidate in translation[word]:
                translation_candidates.append(translation_candidate)
            candidate_array.append(translation_candidates)
        except KeyError:
            continue
    combinations = np.array((np.meshgrid(*candidate_array))).T.reshape(-1,len(to_translate_list))
    highestprob = 0
    print("trigram")
    for combination in combinations.tolist():
        total_prob = 0
        for index in range(len(combination)-2):
            prob = trigram_probs[combination[index]][combination[index+1]][combination[index+2]]
            if prob == 0:
                try:
                    prob = 0.001*(bigram_probs[combination[index]][combination[index+1]] + bigram_probs[combination[index+1]][combination[index+2]])
                except KeyError:
                    pass
            total_prob += prob
        if total_prob > highestprob:
            highestprob = total_prob
            best_sentance_list = combination
    return best_sentance_list


def viterbi_translate(filename, to_translate_list, translation, total_words):
    max_l = 0
    for word in to_translate_list:
        l = len(translation[word])
        if l > max_l:
            max_l = l

    decoder = ViterbiTrigramDecoder(len(to_translate_list), max_l, total_words)
    decoder.init_a(filename)
    decoder.init_b(to_translate_list, translation)
    translation_list = decoder.viterbi(to_translate_list, max_l, translation)


def bigram_translate(bigram_probs, to_translate_list):
    highestprob = 0
    highestprob = float(highestprob)
    best_word1 = translation[to_translate_list[0]][0]
    best_word2 = translation[to_translate_list[1]][0]
    for candidate1 in translation[to_translate_list[0]]:
        for candidate2 in translation[to_translate_list[1]]:
            prob = bigram_probs[candidate1][candidate2]
            if prob > highestprob:
                highestprob = prob
                best_word1 = candidate1
                best_word2 = candidate2
    return [best_word1, best_word2]

def direct_translate(sw_word_list, filename_t):
    for sw_word in sw_word_list:
        get_translation(sw_word, filename_t)
    direct_translation_list = []
    for sw_word in sw_word_list:
        direct_translation_list.append(translation[sw_word][0])
    return direct_translation_list

def get_probabilites(filename):
    reader = Reader()
    reader.read_model(filename)
    return reader.bigram_prob, reader.trigram_prob, reader.unique_words

def main():
    parser = argparse.ArgumentParser(description='DD_1418_projekt')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file from which to get/store the translations')
    parser.add_argument('--prob', '-p', type=str,  required=True, help='file from which to get the probabilites')
    arguments = parser.parse_args()
    
    read_file(arguments.file) # Skapar fil alternativt läser från exiserande fil med översättningar

    bigram_probs, trigram_probs, total_words = get_probabilites(arguments.prob)  # Hämtar bigram och trigram sannolikheter från Reader.py

    while True:
        to_translate = input("Skriv \"quit\" för att avsluta programmet\nSkriv in vad du vill översätta: ")
        to_translate_lower = to_translate.lower()
        if to_translate_lower == "quit":
            print("Tack för denna gång, välkommen tillbaka")
            break
        to_translate_list = to_translate_lower.split()

        direct_translation_list = direct_translate(to_translate_list, arguments.file)  # Direktöversätter
        print("Direktöversättning: " + " ".join(direct_translation_list))

        trigram_translation_list = n_gram_translate(bigram_probs, trigram_probs, to_translate_list, direct_translation_list)  # Skapar en förbättrad översättning
        print("Förbättrad översättning: " + ' '.join(trigram_translation_list))

        #viterbi_translate_list = viterbi_translate(arguments.prob, to_translate_list, translation, total_words)
        #print("Viterbi-översättning: " + ' '.join(viterbi_translate_list))


if __name__ == "__main__":
    main()