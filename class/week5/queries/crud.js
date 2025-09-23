// --- CREATE ---
db.student.insertOne({
  studentId: 101,
  name: "Aarav Patel",
  age: 21,
  email: "aarav.patel@example.com",
  major: "Computer Science",
  minor: "Statistics",
  year: 3,
  gpa: 8.85
});


// --- READ ---
db.student.find({ "major": "Physics" });


// --- UPDATE ---
db.student.updateOne(
  { "studentId": 42 },
  { 
    $set: { 
      "minor": "Data Science",
      "gpa": 9.1 
    } 
  }
);

// --- DELETE ---
db.student.deleteOne({ "studentId": 78 });