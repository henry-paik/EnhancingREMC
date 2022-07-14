ID_TO_LABEL = {0: 'NA', 1: '/location/location/contains', 2: '/people/person/nationality', 3: '/people/person/place_of_birth', 4: '/people/deceasedperson/place_of_death', 5: '/people/person/place_lived', 6: '/business/company/founders', 7: '/people/person/ethnicity', 8: '/location/neighborhood/neighborhood_of', 9: '/business/person/company', 10: '/location/administrative_division/country', 11: '/business/company/place_founded', 12: '/location/country/administrative_divisions', 13: '/location/country/capital', 14: '/people/person/children', 15: '/people/person/religion', 16: '/business/company/majorshareholders', 17: '/people/ethnicity/geographic_distribution', 18: '/business/location', 19: '/business/company/advisors', 20: '/location/us_county/county_seat', 21: '/film/film/featured_film_locations', 22: '/time/event/locations', 23: '/location/region/capital', 24: '/people/deceasedperson/place_of_burial'}
LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}
print(LABEL_TO_ID)
MINOR_LABEL_IDS = [19, 21, 22, 24, 17, 20]
MINOR_LABELS = [ID_TO_LABEL[i] for i in MINOR_LABEL_IDS]

REF_SENT =  {20: 'The relation is "location of us county and county seat". location: the marking out of the boundaries, or identifying the place or site of, a piece of land, according to the description given in an entry, plan, map, etc. county: a region created by territorial division for the purpose of local government. county seat: the town or city that is the seat of government for a county.',
17: 'The relation is "people ethnicity and geographic distribution". ethnicity: an ethnic quality or affiliation resulting from racial or cultural ties. geographic: determined by geography. distribution: the spatial or geographic property of being scattered about over a range, area, or volume.',
24: 'The relation is "people deceased and place of burial". deceased: no longer alive, dead. place: an area, somewhere within an area. burial: the act of burying, interment, concealing something under the ground.',
21: 'The relation is "film and featured film locations". film: cinema; movies as a group. feature: to ascribe the greatest importance to something within a certain context. location: the marking out of the boundaries, or identifying the place or site of, a piece of land, according to the description given in an entry, plan, map, etc.',
19: 'The relation is "business company and advisor". business: commercial, industrial, or professional activity. company: any business, whether incorporated or not, that manufactures or sells products (also known as goods), or provides services as a commercial venture. advisor: one who offers advice, an expert who gives advice.',
22: 'The relation is "event and location". event: something that happens at a given place and time. location: the marking out of the boundaries, or identifying the place or site of, a piece of land, according to the description given in an entry, plan, map, etc.',}



