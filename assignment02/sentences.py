#!/usr/bin/env python

# Include the list of subject pronouns in your grammar:
# http://www.wordreference.com/conj/FrVerbs.aspx

if __name__ == '__main__':
    sentences = [
        # The basic word order in French is Subject-Verb-Object, 
        # as in English:
        "Je regarde la télévision.",
        "Le chat mange le poisson.",
        # Just as in English, the subject must agree 
        # with the verb in number and person:
        "Tu regardes la télévision.",
        "Il regarde la télévision.",
        "Nous regardons la télévision.",
        "Vous regardez la télévision.",
        "Ils regardent la télévision.",
        # A definite noun phrase in French follows a similar 
        # order as in English (article + noun). However, the article 
        # must agree with the noun in number and grammatical gender. 
        # Grammatical gender is a more-or-less arbitrary categorization 
        # of nouns into either masculine or feminine.
        # 
        # As you can see, there is no distinction in the plural
        # between masculine or feminine.
        "Le chat.",
        "La télévision.",
        "Les chats.,",
        "Les télévisions.",
        # Proper names that do not take articles:
        "Jackie",
        "Montréal",
        # When a pronoun is a direct object of the verb, 
        # they precede the verb:
        "Il la regarde.",

        # !! adjective rules !!
        # 1) Adjectives typically follow the noun that 
        # they modify in a noun phrase:
        "Le chat noir",
        "Le chat heureux",
        # 2) However, other adjectives precede the noun:
        "Le beau chat",
        "Le joli chat",
        # 3) Yet others may precede OR follow the noun, 
        # though the meaning usually changes slightly:
        "La dernière semaine",
        "La semaine dernière",
        # 4) In addition, adjectives must agree with the 
        # noun that they modify in number and gender:
        # Note that adjectives do distinguish masculine 
        # from feminine in the plural.
        # 
        # Find several adjectives of each of the three 
        # classes above, and incorporate them into your grammar.
        # http://french.about.com/od/grammar/a/adjectives.htm
        # http://french.about.com/od/grammar/a/adjectives_4.htm
        "Les chats noirs",
        "La télévision noire",
        "Les télévisions noires"
        ]
    
    for sentence in sentences:
        print(sentence)