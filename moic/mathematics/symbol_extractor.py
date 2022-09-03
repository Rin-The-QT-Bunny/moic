from tqdm import tqdm
from aluneth.mathematics.symbol_types import Equation, Function, Expression, Variable, Value, Rational
import sympy
import re

def type_correct_sentence(sentence):
    expressions = extract_formal_elements(sentence, cast = False)
    output_sentence = ""
    last_pointer = 0
    for expression in expressions:
        e = str(expression)
        loc_expression = sentence.index(e)
        output_sentence += sentence[last_pointer:loc_expression]
        output_sentence +=  cast_formal_element(expression).__class__.__name__ + " "
        last_pointer = loc_expression + 1 + len(e)
    return output_sentence

def is_numeric(string):
    return all([x.isnumeric() or x == "." for x in string] + [string.count(".") <= 1])


def extract_formal_elements_as_annotations(question):
    pattern = "\$f\[(.+?)\]"
    return re.findall(pattern, question)


def extract_formal_elements(question, cast=True):
    # split on punctuation unless it is immediately preceded and followed by a number (indicating it is a decimal)
    split_on_punctuation = "***".join(
        [
            string
            for string in re.split("(?<![0-9])[.,;:?]|[.,;:?](?![0-9])", question)
            if len(string) > 0 and not string.isspace()
        ]
    )
    # TODO: use a more sophisticated mechanism (CFG?) to math expressions, equations, etc... this could account for variables names that have length greater than 1
    split_on_words = [
        string
        for string in re.split("[A-Za-z]\w+|\*\*\*", split_on_punctuation)
        if len(string) > 0 and not string.isspace()
    ]
    # strip trailing or leading whitespace
    formal_elements = [string.strip() for string in split_on_words]
    # filter for the special case where the letter "a" gets included at the end of a formal element
    formal_elements = [
        f if len(re.findall("[0-9A-Za-z\)](\sa)", f)) < 1 else f.split(" a")[0]
        for f in formal_elements
    ]
    # cast types
    if cast:
        formal_elements = [cast_formal_element(f) for f in formal_elements]
    return formal_elements


def cast_formal_element(f):
    try:
        x = sympy.sympify(f)
        if type(x) == sympy.core.numbers.Rational:
            return Rational(str(x))
        elif issubclass(type(x), sympy.core.numbers.Number):
            return Value(str(x))
        elif type(x) == sympy.core.symbol.Symbol:
            return Variable(f)
        else:
            return Expression(f)
    except:
        if "=" in f:
            try:
                return Function(f)
            except:
                return Equation(f)


def guess_until_problem_solved(env, question, answer, verbose=False, max_episode_index=1000):
    episode_i = 0
    graph_guessed_correctly = False
    encoded_question, _ = env.reset_from_text(question, answer)
    print(f"\nquestion: {env.decode_question(encoded_question)}")
    while not graph_guessed_correctly and episode_i < max_episode_index:
        encoded_question, _ = env.reset_from_text(question, answer)
        done = False
        step_i = 0
        if verbose:
            print(f"episode: {episode_i}")
        while not done:
            action_index = env.sample_masked_action_index()
            observation, reward, done, info = env.step(action_index)
            if verbose:
                if "lookup_value(solve_system(append_to_empty_list('p_0')),Variable('b'))" in info['raw_observation']:
                    print()
                print(f"\t\tS': {info['raw_observation']}, R: {reward}, done: {done}")
            if reward == 1:
                graph_guessed_correctly = True
            step_i += 1
        episode_i += 1
    assert graph_guessed_correctly
    print(f'graph: {info["raw_observation"].split(";")[1]}')
    print(f"{episode_i} trials taken to guess: {question}")


def filter_univariate(examples):
    univariate_examples = []
    for example_dict in examples:
        question = example_dict['question']
        formal_elements = extract_formal_elements(question, cast=False)
        function = formal_elements[0]
        num_vars = len([ch for ch in set(function) if ch.isalpha()])
        if num_vars == 1:
            univariate_examples.append(example_dict)
    return univariate_examples