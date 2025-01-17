HOVER_PROGRAM_FC = '''Generate a Python-like program that describes the reasoning steps required to verify the claim step-by-step. 

You can use three functions in the program:
1. `Question()`: To answer a specific question.
2. `Verify()`: To verify a single fact or claim.
3. `Predict()`: To predict the veracity label based on verified facts.

Rules:
- Each program must only answer the specific claim provided.
- Avoid any repeated or redundant lines in the output.
- The program must be concise and contain only the necessary steps to verify the claim.
- Stop after providing the required program. Do not generate additional programs or repeat the process.

**Examples**:
# Example 1
# The claim is that Howard University Hospital and Providence Hospital are both located in Washington, D.C.
def program():
    fact_1 = Verify("Howard University Hospital is located in Washington, D.C.")
    fact_2 = Verify("Providence Hospital is located in Washington, D.C.")
    label = Predict(fact_1 and fact_2)
# Example 2
# The claim is that WWE Super Tuesday took place at an arena that currently goes by the name TD Garden.
def program():
    answer_1 = Question("Which arena the WWE Super Tuesday took place?")
    fact_1 = Verify(f"{answer_1} currently goes by the name TD Garden.")
    label = Predict(fact_1)
# Example 3
# The claim is that Talking Heads, an American rock band that was "one of the most critically acclaimed bands of the 80's" is featured in KSPN's AAA format.
def program():
    fact_1 = Verify("Talking Heads is an American rock band that was 'one of the most critically acclaimed bands of the 80's'.")
    fact_2 = Verify("Talking Heads is featured in KSPN's AAA format.")
    label = Predict(fact_1 and fact_2)
# Example 4
# The claim is that An IndyCar race driver drove a Formula 1 car designed by Peter McCool during the 2007 Formula One season.
def program():
    answer_1 = Question("Which Formula 1 car was designed by Peter McCool during the 2007 Formula One season?")
    fact_1 = Verify(f"An IndyCar race driver drove the car {answer_1}.")
    label = Predict(fact_1)
# Example 5
# The claim is that Gina Bramhill was born in a village. The 2011 population of the area that includes this village was 167,446.
def program():
    answer_1 = Question("Which village was Gina Bramhill born in?")
    fact_1 = Verify(f"The 2011 population of the area that includes {answer_1} was 167,446.")
    label = Predict(fact_1)
# Example 6
# The claim is that Don Ashley Turlington graduated from Saint Joseph's College, a private Catholic liberal arts college in Standish.
def program():
    fact_1 = Verify("Saint Joseph's College is a private Catholic liberal arts college is located in Standish.")
    fact_2 = Verify(f"Don Ashley Turlington graduated from Saint Joseph's College.")
    label = Predict(fact_1 and fact_2)
# Example 7
# The claim is that Gael and Fitness are not published in the same country.
def program():
    answer_1 = Question("Which country was Gael published in?")
    answer_2 = Question("Which country was Fitness published in?")
    fact_1 = Verify(f"{answer_1} and {answer_2} are not the same country.")
    label = Predict(fact_1)
# Example 8
# The claim is that Blackstar is the name of the album released by David Bowie that was recorded in secret.
def program():
    fact_1 = Verify("David Bowie released an album called Blackstar.")
    fact_2 = Verify("David Bowie recorded an album in secret.")
    label = Predict(fact_1 and fact_2)
    
# New Question:
# The claim is that [[CLAIM]]
def program():'''

FEVEROUS_PROGRAM_FC = '''Generate a python-like program that describes the reasoning steps required to verify the claim step-by-step. You can call three functions in the program: 1. Question() to answer a question; 2. Verify() to verify a simple claim; 3. Predict() to predict the veracity label. Several examples are given as follows.

# The claim is that In 1959, former Chilean boxer Alfredo Cornejo Cuevas (born June 6, 1933) won the gold medal in the welterweight division at the Pan American Games (held in Chicago, United States, from August 27 to September 7) in Chicago, United States, and the world amateur welterweight title in Mexico City.
def program():
    fact_1 = Verify("Alfredo Cornejo Cuevas was born in June 6, 1933.")
    fact_2 = Verify("Alfredo Cornejo Cuevas won the gold medal in the welterweight division at the Pan American Games in 1959.")
    fact_3 = Verify("The Pan American Games in 1959 was held in Chicago, United States, from August 27 to September 7.")
    fact_4 = Verify("Alfredo Cornejo Cuevas won the world amateur welterweight title in Mexico City.")
    label = Predict(fact_1 and fact_2 and fact_3 and fact_4)

# The claim is that The Footwork FA12, which was intended to start the season, finally debuted at the San Marino Grand Prix, a Formula One motor race held at Imola on 28 April 1991.
def program():
    fact_1 = Verify("The Footwork FA12, which was intended to start the season.")
    fact_2 = Verify("The Footwork FA12 finally debuted at the San Marino Grand Prix.")
    fact_3 = Verify("The San Marino Grand Prix was a Formula One motor race held at Imola on 28 April 1991.")
    label = Predict(fact_1 and fact_2 and fact_3)

# The claim is that SkyHigh Mount Dandenong (formerly Mount Dandenong Observatory) is a restaurant located on top of Mount Dandenong, Victoria, Australia.
def program():
    fact_1 = Verify("SkyHigh Mount Dandenong is a restaurant located on top of Mount Dandenong, Victoria, Australia.")
    fact_2 = Verify("SkyHigh Mount Dandenong is formerly known as Mount Dandenong Observatory.")
    label = Predict(fact_1 and fact_2)

# The claim is that Before the first Europeans arrived or copra companies leased it, Maupihaa was home to Inca's in ancient times.
def program():
    fact_1 = Verify("Maupihaa was home to Inca's in ancient times.")
    fact_2 = Verify("Maupihaa was home to Inca's before the first Europeans arrived or copra companies leased it.")
    label = Predict(fact_1 and fact_2)

# The claim is that Shulin, a 33.1288 km (12.7911 sq mi) land located in New Taipei City, China, a country in East Asia, has a total population of 183,946 in December 2018.
def program():
    fact_1 = Verify("Shulin is a 33.1288 km (12.7911 sq mi) land located in New Taipei City, China.")
    fact_2 = Verify("Shulin has a total population of 183,946 in December 2018.")
    label = Predict(fact_1 and fact_2)

# The claim is that Sumo wrestler Toyozakura Toshiaki committed match-fixing, ending his career in 2011 that started in 1989.
def program():
    fact_1 = Verify("Toyozakura Toshiaki ended his career in 2011 that started in 1989.")
    fact_2 = Verify("Toyozakura Toshiaki is a Sumo wrestler.")
    fact_3 = Verify("Toyozakura Toshiaki committed match-fixing.")
    label = Predict(fact_1 and fact_2 and fact_3)

# The claim is that In 1959, former Chilean boxer Alfredo Cornejo Cuevas (born June 6, 1933) won the gold medal in the welterweight division at the Pan American Games (held in Chicago, United States, from August 27 to September 7) in Chicago, United States, and the world amateur welterweight title in Mexico City.
def program():
    fact_1 = Verify("Alfredo Cornejo Cuevas is a former Chilean boxer.")
    fact_2 = Verify("Alfredo Cornejo won the gold medal in the welterweight division at the Pan American Games.")
    fact_3 = Verify("The Pan American Games was held in Chicago, United States, from August 27 to September 7.")
    fact_4 = Verify("Alfredo Cornejo won the world amateur welterweight title in Mexico City.")
    label = Predict(fact_1 and fact_2 and fact_3 and fact_4)

# The claim is that Adductor hiatus is associated with nine structures, seven of which enter and leave through hiatus.
def program():
    fact_1 = Verify("Adductor hiatus is associated with nine structures.")
    fact_2 = Verify("Seven of the nine structures associated with Adductor hiatus enter and leave through hiatus.")
    label = Predict(fact_1 and fact_2)

# The claim is that Ifor Bowen Lloyd was educated at Winchester (an independent boarding school for boys in the British public school tradition) and Exeter College, Oxford where he was a member of the Library Committee of the Oxford Union Society, as well as, received a BA in Modern History in 1924.
def program():
    fact_1 = Verify("Ifor Bowen Lloyd was educated at Winchester and Exeter College, Oxford.")
    fact_2 = Verify("Winchester is an independent boarding school for boys in the British public school tradition.")
    fact_3 = Verify("While at Oxford, Ifor Bowen Lloyd was a member of the Library Committee of the Oxford Union Society.")
    fact_4 = Verify("Ifor Bowen Lloyd received a BA in Modern History in 1924 at Oxford.")
    label = Predict(fact_1 and fact_2 and fact_3 and fact_4)

# The claim is that In the 2001 Stanley Cup playoffs Eastern Conference Semifinals Devils' Elias scored and Maple Leafs' left Devils player Scott Neidermayer hurt.
def program():
    fact_1 = Verify("In the 2001 Stanley Cup playoffs Eastern Conference Semifinals Devils' Elias scored.")
    fact_2 = Verify("Maple Leafs' left Devils player Scott Neidermayer hurt.")
    label = Predict(fact_1 and fact_2)

# The claim is that Teldenia helena is a moth first described in 1967 by Wilkinson.
def program():
    fact_1 = Verify("Teldenia helena is a moth.")
    fact_2 = Verify("Teldenia helena was first described by Wilkinson in 1967.")
    label = Predict(fact_1 and fact_2)

# The claim is that Born December 30, 1974, William Frick was a dark horse candidate in the Maryland House of Delegates appointment process.
def program():
    fact_1 = Verify("William Frick was born in December 30, 1974.")
    fact_2 = Verify("William Frick was a dark horse candidate in the Maryland House of Delegates appointment process.")
    label = Predict(fact_1 and fact_2)

# The claim is that [[CLAIM]]
def program():'''


class Prompt_Loader:
    def __init__(self) -> None:
        self.hover_program_fc = HOVER_PROGRAM_FC
        self.feverous_program_fc = FEVEROUS_PROGRAM_FC

    def prompt_construction(self, claim, dataset_name):
        template = None
        if dataset_name == 'HOVER':
            template = self.hover_program_fc
        elif dataset_name == 'FEVEROUS':
            template = self.feverous_program_fc
        else:
            raise NotImplementedError

        return template.replace('[[CLAIM]]', claim)