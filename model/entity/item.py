import re

from model.dict.content_dict import ContentDict
from model.entity.base_entity import Entity


class Item(Entity):
    def __init__(self, item_id: str, alias_id: int):
        super().__init__(item_id, alias_id)

    def _try_to_get_property_from_content(self, key) -> object:
        try:
            return self._data['Content'][key]
        except KeyError:
            return None

    @property
    def runtime(self) -> int:
        return self._try_to_get_property_from_content("Runtime")

    @property
    def imdb_rate(self) -> float:
        return self._try_to_get_property_from_content("imdbRating")

    @property
    def year(self) -> int:
        return self._try_to_get_property_from_content("Year")

    @property
    def awards(self) -> dict:
        return self._try_to_get_property_from_content("Awards")

    @property
    def genres(self) -> dict:
        return self._try_to_get_property_from_content("Genre")

    @property
    def countries(self) -> dict:
        return self._try_to_get_property_from_content("Country")

    @property
    def languages(self) -> dict:
        return self._try_to_get_property_from_content("Language")

    @property
    def directors(self) -> dict:
        return self._try_to_get_property_from_content("Director")

    @property
    def actors(self) -> dict:
        return self._try_to_get_property_from_content("Actors")

    @property
    def metascore(self) -> int:
        return self._try_to_get_property_from_content("Metascore")

    def add_content(self, content: dict, content_dict: ContentDict):
        # with open('inputs/stopwords.txt') as f:
        #     stopwords_list = f.read().splitlines()

        def get_runtime() -> int:
            item_runtime = content['Runtime']
            if item_runtime == 'N/A':
                return 0
            elif ' h ' not in item_runtime:
                return int(item_runtime.replace(" min", ""))
            else:
                data = item_runtime.replace(
                    " h ", ",").replace(" min", "").split(",")
                return int(data[0]) * 60 + int(data[1])

        def get_imdb_rating():
            imdb_rating = content['imdbRating']
            if imdb_rating != 'N/A':
                return float(imdb_rating)
            else:
                return 0

        def filter_lists(key):
            payload = {}
            if content[key] != 'N/A':
                if key == "Director" or key == "Actors":
                    s = content[key].replace(", ", ",").lower().split(',')
                else:
                    s = content[key].replace(" ", "").lower().split(',')
                for g in s:
                    content_dict.append(g)
                    payload.update({str(g): g})
            return payload

        def stract_plot(plot):
            plot = plot.lower()
            plot = re.sub("[^a-zA-Z]+", " ", plot)

            return [word for word in plot.split(" ") if word not in stopwords_list]

        def awards_structure(awards):
            def extract_values(regex, string):
                value = re.findall(regex, string)
                if not len(value):
                    return 0
                return sum([int(s_find) for s_find in re.findall(r'\b\d+\b', value[0])])

            s = awards.lower()

            structure = {
                "Win": extract_values(r'\b\d+\b win', s),
                "Nomination": extract_values(r'\b\d+\b nomination', s),
                "BAFTA Won": extract_values(r'won \b\d+\b bafta', s),
                "BAFTA Nomination": extract_values(r'nominated for \b\d+\b bafta', s),
                "Oscar Won": extract_values(r'won \b\d+\b oscar', s),
                "Oscar Nomination": extract_values(r'nominated for \b\d+\b oscar', s),
                "Golden Won": extract_values(r'won \b\d+\b golden', s),
                "Golden Nomination": extract_values(r'nominated for \b\d+\b golden', s),
                "Primetime Emmy Nomination": extract_values(r'nominated for \b\d+\b primetime', s),
                "Primetime Emmy Won": extract_values(r'won \b\d+\b primetime', s)
            }

            # This lines asserts if all regex are right
            # assert sum([int(s_find) for s_find in re.findall(r'\b\d+\b', s)]) == sum([d for d in structure.values()])

            return structure

        def get_imdb_rate():
            if content["imdbVotes"] == "N/A":
                return 0
            return int(content["imdbVotes"].replace(",", "").replace(".", ""))

        def get_metascore():
            if content["Metascore"] == "N/A":
                return 0
            return int(content["Metascore"])

        self._data.update(
            {
                "Content": {
                    "Title": content["Title"],
                    "Year": int(content["Year"]),
                    "Rated": content["Rated"],
                    "Released": content["Released"],
                    "Runtime": get_runtime(),
                    "Genre": filter_lists("Genre"),
                    "Director": filter_lists("Director"),
                    "Writer": content["Writer"],
                    "Actors": content["Actors"],
                    "Plot": content["Plot"],
                    "Language": filter_lists("Language"),
                    "Country": filter_lists("Country"),
                    "Awards": awards_structure(content["Awards"]),
                    "Poster": content["Poster"],
                    "Metascore": get_metascore(),
                    "imdbRating": get_imdb_rating(),
                    "imdbVotes": get_imdb_rate(),
                    "imdbID": content["imdbID"],
                    "Type": content["Type"],
                    "Response": eval(content["Response"])
                }
            }
        )
