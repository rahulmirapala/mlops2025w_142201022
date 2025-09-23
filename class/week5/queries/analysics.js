// ANALYSIS 1: Count Students per Year
[
  {
    '$group': {
      '_id': '$year', 
      'numberOfStudents': {
        '$sum': 1
      }
    }
  }, {
    '$sort': {
      '_id': 1
    }
  }
]


// ANALYSIS 2: Average GPA per Major
[
  {
    '$group': {
      '_id': '$major', 
      'averageGPA': {
        '$avg': '$gpa'
      }
    }
  }, {
    '$sort': {
      'averageGPA': -1
    }
  }
]


// ANALYSIS 3: Top 5 Computer Science Students by GPA
[
  {
    '$match': {
      'major': 'Computer Science'
    }
  }, {
    '$sort': {
      'gpa': -1
    }
  }, {
    '$limit': 5
  }
]