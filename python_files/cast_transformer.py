import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CastTransformer(TransformerMixin, BaseEstimator):
    # BaseEstimator generates the get_params() and set_params() methods that all Pipelines require
    # TransformerMixin creates the fit_transform() method from fit() and transform()

    def __init__(self):
        pass

    def fit(self, X, y=None):
        #         for feature_number in range(X.shape[1]):
        #         def top_k(feature, k):
        #         feature_top_k = np.partition(X[:,0], -k)[-k:]
        #         self.popular_actor_list = np.sort(feature_top_k)[::-1]
        return self

    def transform(self, X, y=None):
        #         popular_actor_list = ['Nicolas Cage',
        #   'Johnny Depp',
        #   'Ajith Kumar',
        #   'Bruce Willis',
        #   'Tom Cruise',
        #   'Denzel Washington',
        #   'Tom Hanks',
        #   'Adam Sandler',
        #   'Robert De Niro',
        #   'John Travolta',
        #   'Eddie Murphy',
        #   'John Wayne',
        #   'Keanu Reeves',
        #   'Arnold Schwarzenegger',
        #   'Ben Affleck',
        #   'Jeff Bridges',
        #   'George Clooney',
        #   'Harrison Ford',
        #   'Matt Damon',
        #   'Liam Neeson',
        #   'Kevin Costner',
        #   'Mark Wahlberg',
        #   'Clint Eastwood',
        #   'Jason Statham',
        #   'Mel Gibson',
        #   'Jackie Chan',
        #   'Sylvester Stallone',
        #   'Meryl Streep',
        #   'Ben Stiller',
        #   'Christian Bale']
        popular_director_list = [
            "Steven Spielberg",
            "Woody Allen",
            "Ridley Scott",
            "Alfred Hitchcock",
            "Clint Eastwood",
            "Steven Soderbergh",
            "Oliver Stone",
            "Martin Scorsese",
            "Robert Rodriguez",
            "John Carpenter",
            "Tim Burton",
            "John Ford",
            "Wes Craven",
            "Francis Ford Coppola",
            "Ron Howard",
            "Roland Emmerich",
            "Robert Zemeckis",
            "Michael Bay",
            "Billy Wilder",
            "Bobby Farrelly",
            "Renny Harlin",
            "Sam Raimi",
            "Peter Jackson",
            "Walter Hill",
            "Tony Scott",
            "Richard Donner",
            "Brian De Palma",
            "Barry Levinson",
            "Spike Lee",
            "John Landis",
        ]
        popular_producer_list = [
            "Joel Silver",
            "Jerry Bruckheimer",
            "Brian Grazer",
            "Neal H. Moritz",
            "Tim Bevan",
            "Luc Besson",
            "John Davis",
            "Michael Bay",
            "Scott Rudin",
            "Arnon Milchan",
            "Danny DeVito",
            "Clint Eastwood",
            "Roger Birnbaum",
            "Ridley Scott",
            "Charles Roven",
            "Steve Golin",
            "Ronnie Screwvala",
            "Albert R. Broccoli",
            "Lawrence Gordon",
            "Walt Disney",
            "Tom Rosenberg",
            "Gale Anne Hurd",
            "Alfred Hitchcock",
            "Lawrence Bender",
            "Kathleen Kennedy",
            "Nick Wechsler",
            "Steven Spielberg",
            "John Hughes",
            "Judd Apatow",
            "Robert Rodriguez",
        ]
        popular_prod_company_list = [
            "Universal Pictures",
            "Paramount Pictures",
            "Twentieth Century Fox Film Corporation",
            "Columbia Pictures",
            "New Line Cinema",
            "Walt Disney Pictures",
            "Warner Bros.",
            "Metro-Goldwyn-Mayer (MGM)",
            "United Artists",
            "Columbia Pictures Corporation",
            "Miramax Films",
            "TriStar Pictures",
            "Village Roadshow Pictures",
            "DreamWorks SKG",
            "Fox Searchlight Pictures",
            "Summit Entertainment",
            "Touchstone Pictures",
            "Orion Pictures",
            "Lions Gate Films",
            "BBC Films",
            "Lionsgate",
            "Imagine Entertainment",
            "The Weinstein Company",
            "Regency Enterprises",
            "RKO Radio Pictures",
            "Hollywood Pictures",
            "Lakeshore Entertainment",
            "StudioCanal",
            "Lucasfilm",
            "Dimension Films",
        ]
        prod_country_list = [
            "United States of America",
            "United Kingdom",
            "India",
            "France",
            "Germany",
            "Canada",
            "Australia",
            "Russia",
            "Japan",
            "Italy",
            "China",
            "Spain",
            "Ireland",
            "Hong Kong",
            "South Korea",
            "Mexico",
            "Belgium",
            "New Zealand",
            "Czech Republic",
            "Denmark",
            "Netherlands",
            "Switzerland",
            "Singapore",
            "Israel",
            "Brazil",
            "Thailand",
            "United Arab Emirates",
            "Norway",
            "Sweden",
            "Finland",
        ]
        #         actor_transformed = []
        prod_country_transformed = []
        director_transformed = []
        producer_transformed = []
        production_company_transformed = []
        #         actor_np = np.zeros((30,), dtype=int)

        for prod_country in X[:, 0]:
            prod_country_np = np.zeros((30,), dtype=int)
            if prod_country in prod_country_list:
                index = prod_country_list.index(prod_country)
                prod_country_np[index] = 1
            prod_country_transformed.append(list(prod_country_np))
        for director in X[:, 1]:
            director_np = np.zeros((30,), dtype=int)
            if director in popular_director_list:
                index = popular_director_list.index(director)
                director_np[index] = 1
            director_transformed.append(list(director_np))
        for producer in X[:, 2]:
            producer_np = np.zeros((30,), dtype=int)
            if producer in popular_producer_list:
                index = popular_producer_list.index(producer)
                producer_np[index] = 1
            producer_transformed.append(list(producer_np))
        for pc in X[:, 3]:
            production_company_np = np.zeros((30,), dtype=int)
            if pc in popular_prod_company_list:
                index = popular_prod_company_list.index(pc)
                production_company_np[index] = 1
            production_company_transformed.append(list(production_company_np))
        return np.concatenate(
            (prod_country_transformed, director_transformed, producer_transformed, production_company_transformed),
            axis=1,
        )
