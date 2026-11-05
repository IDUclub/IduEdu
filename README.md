# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/IDUclub/IduEdu/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                  |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------ | -------: | -------: | -------: | -------: | ------: | --------: |
| iduedu/\_\_init\_\_.py                                |        4 |        0 |        0 |        0 |    100% |           |
| iduedu/\_api.py                                       |       30 |       13 |        2 |        0 |     53% |66-69, 84-93 |
| iduedu/\_numba/\_\_init\_\_.py                        |        1 |        0 |        0 |        0 |    100% |           |
| iduedu/\_numba/components.py                          |       87 |        3 |       34 |        2 |     96% | 10, 16-17 |
| iduedu/\_numba/csr.py                                 |       39 |        1 |        0 |        0 |     97% |        47 |
| iduedu/\_numba/shortest\_paths.py                     |       76 |        0 |       36 |        2 |     98% |34-\>33, 39-\>38 |
| iduedu/\_version.py                                   |        1 |        0 |        0 |        0 |    100% |           |
| iduedu/config.py                                      |      163 |        3 |       56 |        4 |     97% |90, 251, 290, 342-\>exit |
| iduedu/constants/\_\_init\_\_.py                      |        1 |        0 |        0 |        0 |    100% |           |
| iduedu/constants/highway\_enums.py                    |       28 |        0 |        0 |        0 |    100% |           |
| iduedu/constants/network\_enums.py                    |        9 |        0 |        0 |        0 |    100% |           |
| iduedu/constants/transport\_specs.py                  |      122 |        2 |       54 |        2 |     98% |   95, 108 |
| iduedu/graph/\_\_init\_\_.py                          |        9 |        0 |        0 |        0 |    100% |           |
| iduedu/graph/adapters.py                              |      113 |       22 |       34 |        8 |     77% |16, 97, 109-111, 115-116, 125-126, 143-144, 150-157, 210-216 |
| iduedu/graph/adjacency.py                             |       70 |        5 |       26 |        4 |     91% |43, 106, 113, 126-127 |
| iduedu/graph/components.py                            |       65 |       10 |       16 |        5 |     81% |16, 82-88, 105, 114, 123, 142-143, 175-\>177, 180, 183 |
| iduedu/graph/editors.py                               |      377 |       44 |      180 |       42 |     84% |61, 63, 65, 67, 72, 74, 83, 85, 87, 92, 98, 100, 120-\>131, 125-\>131, 128, 156, 169, 202, 208-\>213, 245, 247, 251, 253, 255, 257, 274, 276, 286, 375, 409, 413-414, 441, 609, 619-621, 624-630, 633-\>636, 639-\>585, 641-\>585, 648-656, 658-\>663, 664-\>677, 721 |
| iduedu/graph/graph\_inputs.py                         |       74 |       22 |       52 |       22 |     65% |35, 37, 39, 41, 45, 47, 49, 53, 55, 66, 90, 92, 95, 97, 101, 105, 107, 109, 112, 116, 121, 123 |
| iduedu/graph/io.py                                    |      149 |       32 |       52 |       12 |     72% |41-48, 83, 121-122, 175-176, 178-\>181, 212, 218, 223, 225, 232-234, 264-\>263, 270, 276, 282, 285-286, 290-296 |
| iduedu/graph/nx\_utils.py                             |      173 |       21 |       54 |       14 |     83% |34, 86-87, 92-\>95, 99-\>101, 122-\>127, 125-126, 127-\>121, 130-131, 154-159, 165-166, 198-199, 230-\>239, 281, 290, 295, 297-\>300, 302-\>315, 317 |
| iduedu/graph/routing.py                               |      249 |        7 |       86 |        8 |     96% |63, 65, 67, 69, 72, 393-\>395, 601, 603 |
| iduedu/graph/routing\_result.py                       |      123 |        2 |       38 |        2 |     98% |   18, 112 |
| iduedu/graph/shortest\_paths.py                       |       63 |        8 |       34 |        8 |     84% |55, 57, 59, 64-65, 74, 78, 80, 105-\>104 |
| iduedu/graph/transformers.py                          |       71 |       12 |       30 |       13 |     75% |51, 88, 91-\>100, 97, 124, 164, 166, 168, 171, 182, 184, 187-188, 205-\>208 |
| iduedu/graph/urban\_graph.py                          |      224 |       26 |       36 |        8 |     86% |93-99, 119-\>121, 139, 214, 322-324, 333-335, 355-357, 376, 607-609, 776-778, 795-797, 826, 879, 917, 980, 1061 |
| iduedu/graph/validation.py                            |       89 |       16 |       60 |       16 |     79% |39, 41, 55, 60, 62, 74, 80, 82, 85, 87, 90, 93, 95, 101, 103, 115 |
| iduedu/graph\_builders/\_\_init\_\_.py                |        0 |        0 |        0 |        0 |    100% |           |
| iduedu/graph\_builders/drive\_walk\_builders.py       |      188 |        4 |       52 |        4 |     97% |47, 278, 292, 301 |
| iduedu/graph\_builders/gtfs\_builders.py              |      567 |       63 |      230 |       52 |     85% |62, 64, 69-71, 78, 80-82, 85, 89-90, 92, 101-102, 112-113, 115-\>117, 117-\>119, 128-\>135, 131-\>133, 137-139, 169-\>171, 190, 206-208, 288, 314, 328, 332-336, 342, 368, 391, 400-401, 438-439, 446, 462-\>473, 475, 479-480, 495, 505, 511, 523, 534, 598, 601-\>599, 603, 610, 628, 668, 725-\>736, 755-\>758, 760, 829, 865, 928, 930, 940, 944, 959-961, 970-971 |
| iduedu/graph\_builders/intermodal\_builders.py        |      148 |        8 |       58 |       10 |     91% |29-\>35, 31, 34, 51, 90, 148-\>173, 223, 247-\>249, 311, 330-331 |
| iduedu/graph\_builders/public\_transport\_builders.py |      247 |       12 |       96 |       11 |     93% |42, 47-48, 88, 105, 153, 265, 305, 356, 451, 473-\>482, 477-\>479, 479-\>482, 494-495 |
| iduedu/gtfs/\_\_init\_\_.py                           |        3 |        0 |        0 |        0 |    100% |           |
| iduedu/gtfs/merge.py                                  |      203 |       16 |      100 |       12 |     90% |48, 121-\>123, 166, 211, 235-245, 250-\>252, 259, 262, 322, 355-359, 431, 438 |
| iduedu/gtfs/reader.py                                 |       56 |        6 |       22 |        4 |     87% |53, 56, 59, 78-79, 113 |
| iduedu/gtfs/service\_days.py                          |       68 |        5 |       32 |        5 |     90% |49, 60, 78, 104, 107 |
| iduedu/gtfs/validation.py                             |       57 |        6 |       30 |        6 |     86% |43, 49, 57, 59, 65, 104 |
| iduedu/osm\_pbf/\_\_init\_\_.py                       |        5 |        0 |        0 |        0 |    100% |           |
| iduedu/osm\_pbf/download.py                           |      142 |       28 |       42 |        8 |     77% |81-84, 121-126, 134-135, 137-138, 153-155, 164-167, 205, 210-211, 214-216, 264, 266-\>268, 276 |
| iduedu/osm\_pbf/extract.py                            |      247 |       11 |      132 |        9 |     93% |141, 180-\>184, 217-\>215, 219-\>223, 222, 253-254, 255-\>248, 275-\>274, 303, 312-317 |
| iduedu/osm\_pbf/filters.py                            |      148 |        9 |       54 |        6 |     93% |96, 117, 123, 128, 135, 147-149, 209 |
| iduedu/osm\_pbf/reader.py                             |      356 |       17 |      140 |       24 |     91% |62, 70-\>72, 74-75, 77, 101-104, 114, 131, 141, 152-\>154, 182-\>179, 196-\>210, 201-\>203, 207-\>209, 227-\>229, 242-\>244, 247-\>249, 259-\>256, 290, 312, 329, 354-\>351, 402, 413-\>412, 472-\>470, 502, 512-513 |
| iduedu/overpass/\_\_init\_\_.py                       |        0 |        0 |        0 |        0 |    100% |           |
| iduedu/overpass/cache.py                              |       57 |        4 |       10 |        2 |     91% |63, 68-69, 86 |
| iduedu/overpass/downloaders.py                        |      254 |       43 |       94 |       10 |     82% |107-111, 126-155, 169-\>171, 171-\>174, 224-227, 264, 377, 384, 546, 551-576 |
| iduedu/overpass/parsers.py                            |      746 |       56 |      346 |       51 |     89% |85, 279-\>318, 298-299, 310, 312-314, 316-\>279, 417, 483-484, 517, 519, 532, 538-\>530, 576-577, 662, 674-675, 684, 694-\>712, 715, 741, 753, 760, 840-\>842, 847-\>849, 865-866, 869, 873, 884, 898, 904-\>938, 930-\>936, 931-\>930, 942, 958, 970, 977, 1012, 1035, 1144-1145, 1146-\>1150, 1152, 1207, 1229-1230, 1260-1261, 1264-1265, 1270-\>1268, 1280-\>1279, 1307-1309, 1312, 1327, 1330-1331, 1336, 1340, 1343-1344 |
| **TOTAL**                                             | **5902** |  **537** | **2318** |  **386** | **88%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/IDUclub/IduEdu/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/IDUclub/IduEdu/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/IDUclub/IduEdu/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/IDUclub/IduEdu/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2FIDUclub%2FIduEdu%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/IDUclub/IduEdu/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.