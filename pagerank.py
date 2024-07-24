import os
import random
import re
import sys
import time

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Corpus : Dict mapping a page name to a set of all pages linked to by that page
    # Page : String representing which page the random surfer is currently on
    # damping_factor

    #First, we get the number of page linked to by the current page
    linkNumber = len(corpus[page])
    print(f'Cuurent page : {corpus[page]} number of links : {linkNumber}')
    # Number of sites in corpus
    num_elem = len(corpus)

    probDist = {}

    if linkNumber == 0:
        #If the page has no link, we return a dict of site with all the same probability
        prob = 1/num_elem
        for site in page:
            probDist[site] = prob
        print('Should not be there')
    else:
        for site in corpus:
            if site in corpus[page]:
                probDist[site] = damping_factor*(1/linkNumber) + ((1-damping_factor)/num_elem)
            else:
                probDist[site] = (1-damping_factor)/num_elem

    return probDist

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    

    # Construction of the initial pagerank and probability distribution dictionaries
    PageRankDict = {}
    ProbDistr = {}
    
    for site in corpus.keys():
        ProbDistr[site] = 1/len(corpus)
        PageRankDict[site] = 0

    #Converting corpus to a list so we can pick a radom key
    corpus_list = list(corpus.keys())
    firstPage = random.choice(corpus_list)
    initial_result = ProbDistr
    
    

    i = n

    while i > 0:
        if i == n:
            print(firstPage)
            ProbDistr = transition_model(corpus, firstPage, damping_factor)
            First_result = ProbDistr
            print(ProbDistr)
            PageRankDict[firstPage] += 1/n
            i -= 1
            
        else:
            # Determine which page to give to the model depending on the last PageRank result
            elements = []
            prob = []
            for element, key in ProbDistr.items():
                elements.append(element)
                prob.append(float(key))
            next_page = random.choices(elements, weights = prob, k = 1)
            print('Page #' + str(i) + ' : ' + next_page[0])
            ProbDistr = transition_model(corpus, next_page[0], damping_factor)
            PageRankDict[next_page[0]] += 1/n
            i -= 1
    print(f'init result : #{initial_result}')
    print(f'First result : #{First_result}')
    print(f'Final result : #{PageRankDict}')
    return PageRankDict



def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
