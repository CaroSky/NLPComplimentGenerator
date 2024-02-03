from WordState import WordState

class MarkovChain:
    def __init__(self, sentences):
        # Build the graph for the Markov Chain during initialization
        self.graph = self.build_graph(sentences)

    def build_graph(self, sentences):
        # Initialize an empty graph
        graph = {}

        # Add each sentence to the graph
        for sentence in sentences:
            self.add_sentence_to_graph(sentence, graph)
        return graph

    @staticmethod
    def add_sentence_to_graph(sentence, graph):
        # Split the sentence into words and add a start token
        words = ["#"] + sentence.split()

        # Loop through each word in the sentence
        for i in range(1, len(words)):
            previous_word = words[i - 1]
            current_word = words[i]

            # Build the graph: the keys are words, and the values are dictionaries
            # representing the next words and their frequencies
            graph.setdefault(previous_word, {}).setdefault(current_word, 0)
            graph[previous_word][current_word] += 1

    def get_random_next_word(self, current_word):
        # If the current word is not in the graph, return None
        if current_word not in self.graph:
            return None

        # Use WordState to decide which word comes next based on frequency
        word_state = WordState()
        for next_word, freq in self.graph[current_word].items():
            for _ in range(freq):
                word_state.add_next_word(next_word)

        return word_state.get_next() if word_state.has_next() else None

    def generate(self, max_length=20):
        # Start with the start toke
        current_word = "#"
        sentence = []

        # Generate words until a stopping condition is met
        while True:
            next_word = self.get_random_next_word(current_word)

            # If the next word doesn't exist, or it's a start token, end the sentence
            if not next_word or next_word == "#":
                break

            # Add the next word to the sentence and continue generation
            sentence.append(next_word)
            current_word = next_word

            # If we reach the maximum length, try to find a natural end within a threshold
            if len(sentence) == max_length:
                for _ in range(10):  # Threshold to find a natural ending
                    next_word = self.get_random_next_word(current_word)

                    if not next_word or next_word == "#":
                        break

                    sentence.append(next_word)
                    current_word = next_word
                break

        # Return the generated sentence as a string
        return " ".join(sentence)