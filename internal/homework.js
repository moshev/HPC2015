// Simple node js script that can be used to compile, run and evaluate the result of C++ files

// The tools is designed to be minimalistic and simple. In spite of that it achieves some levels
// of security by sandboxing the compiled C++ files and by limiting the time they have to compile and execute. The sandboxing works in OS X only for the moment.
// Even with sandboxing on, security with this script is designed by keeping in mind that this script will be started within a virtual machine.

//to start external executables
var execSync = require('sync-exec');
//to manipulate files
var fs = require('fs');
//to create directories
var mkdirp = require('mkdirp');
//to check the OS (we enable sandboxing if this is OS X 10.9 or newer)
var os = require('os');

//how many times to run each homework. 10 is default, second cmd arg can change that ("sudo node homework 1 100" will run homework 1, 100 times for each student
var NUM_RUNS = 10;

var args = process.argv;
for (var i = 0; i < args.length; ++i) {
    var arg = args[i];
    if (arg.indexOf("help")>-1) {
        log("\nHOMEWORK.JS is project for automatic homework check, in-house made for Sofia University, FMI, HPC 2015 course\n\
            Here is a simple how-to:\n\
            1. In a folder named ./data you should create 1 folder for each homework assignemnt, named 0, 1, 2, ...\n\
            2. In each of these homework assignemnt folders, you should make folder for each student, which name is the faculty number of the student\n\
            3. In each of these student folders, there should be one .cpp file, that contains the student solution (name of the file doesn't matter, main.cpp is a good name\n\
            4. You should know what is the type of source in those .cpp files. Let say the students had to imeplement function named `fizzBuzz()`\n\
            5. In ./solution<HOMEWORK_NUMBER>.cpp you should have int main() function, that does whatever checks are needed (like, call fizzBuzz and see if the reuslt is as expected). If any of the checks fails, you sould print \"BAD\" and you should call exit(1)\n\
            6. This script will append ./solution<HOMEWORK_NUMBER>.cpp source to the source of each student and will make sure to remove any int main() function that they might have added. It will compile and check if the results were okay and it will report how long it took\n\
            7. This files should be called with `sudo node homework.js <HOMEWORK_NUMBER> <NUMBER_OF_TIME_TO_RUN_EACH_STUDENT_HOMEWORK>. The second arg is optional, with default value = " + NUM_RUNS + ". This script is tested on OS X only.");
        
        process.exit(1)
    }
}

//the index of the homework, that we will test
var HOMEWORK_NUMBER;

//fetching args is lame, but it is what it is
HOMEWORK_NUMBER = args[2];
if (isNaN(HOMEWORK_NUMBER)) {
    log("First arg should be the homework number, got " + HOMEWORK_NUMBER + " instead. This is an error");
    process.exit(1);
}
HOMEWORK_NUMBER = parseFloat(HOMEWORK_NUMBER);
if (HOMEWORK_NUMBER <= 0) {
    log("Homework number should be >= 0, got:" + HOMEWORK_NUMBER + " instead. This is an error");
    proces.exit(1);
}


if (args.length > 3) {
    var tempNumRuns = args[3];
    if (isNaN(tempNumRuns) == false) {
        NUM_RUNS = parseFloat(tempNumRuns);
        if (NUM_RUNS <= 0) {
            log("Second arg that tells how many times to run each test is " + NUM_RUNS + " which is <= 0. This is an error. Script will abort. Please enter something >= 0 ");
            process.exit(1);
        }
    }
}
log("Will run each test for " + tempNumRuns + " times")

log("******************* Starting homework #" + HOMEWORK_NUMBER + " *******************");

var SOLUTIONS_FOLDER = './data/' + HOMEWORK_NUMBER;

var walk    = require('walk');
//****************************************************************************************************************

//this will be used to invoke a compiler
//if you don't have clang++, you wil have to modify that
var COMPILER_START_COMMAND = 'clang++ -Xclang -std=c++14 -O3 -stdlib=libc++ -lpthread';

var USE_SANDBOXING = (os.platform()=="darwin") && parseFloat(os.release()) >= 15;

checkUserDirExists("./data");
checkUserDirExists(SOLUTIONS_FOLDER);

if (USE_SANDBOXING) {
    log("Sandboxing is supported and will be used");
} else {
    log("WARNING! Sandboxing is not supported and will be disabled");
}

//the folder that holds the homeworks for all students
//it should be located in the folder from which this file is started
//if it not exists there, it will be created. In this folder the script will create a folder for each student id.
//In each student id folders, there will be a file named 'sandbox', that has the sandbox configuration for that folder and a folder for each homework.
//In each homework folder, there will be a HOMEWORK_EXECUTABLE_NAME.cpp file (containing the source), HOMEWORK_EXECUTABLE_NAME.bin (containing the executable) and results.txt (containing part of the log that this scripts prints).
var HOMEWORK_FOLDER = "homework_js_temp_folder";

//maximum amount of runtime for each homework
//the same amount is used as max amount of time for clang during homework compilation
var MAX_RUNTIME_MS = 42000;

//the name of the temp file that will be created to hold each of the students homework
//it is good to have some random name, since later we will use "pkill -9 HOMEWORK_EXECUTABLE_NAME" in case execution takes too long
var HOMEWORK_EXECUTABLE_NAME = "3bb005aa-5d34-11e5-885d-feff819cdc9f";

//returns if something is a number
function isNumeric(n) {
    return !isNaN(parseFloat(n)) && isFinite(n);
}

//logs to console
function log(str) {
    var msg ="[" + (new Date()).toString() + "] " + str;
    console.log(msg);
}

//creates file named "sandbox" in the user(student) directory, which has the configuration for running ./sandbox-exec.
//This allows process started with "./sandbox-exec -f ./userdir/sandbox /path/to/process/process.bin" to mess up
//only with files in "path/to/process/" (aka it is sandboxing it)
function createSandboxFile(userFolder) {
    var sandboxFile = userFolder + "/sandbox";
    removeFile(sandboxFile);
    var permissions = "\
    (version 1)\n\
    (deny default)\n\
    \n\
    (deny file-read* file-write*\n\
     (regex \"^/*\"))\n\
    \n\
    (allow file-read-data file-read-metadata \n\
     (regex \"^/Library/Preferences\")\n\
     (regex \"^/Library/PreferencePanes\")\n\
     (regex \"^/usr/share/icu\")\n\
     (regex \"^/usr/share/locale\")\n\
     (regex \"^/System/Library\")\n\
     (regex \"^/usr/lib\")\n\
     (regex \"^/var\")\n\
     (regex \"^/private/var/tmp/mds/\")\n\
     (regex \"^/private/var/tmp/mds/[0-9]+(/|$)\")\n\
     (regex \"" + userFolder + "\"))\n\
     \n\
    (allow process-exec\n\
     (regex \"" + userFolder + "/*\"))\n\
     \n\
    (deny network*)";
    
    try {
        fs.writeFileSync(sandboxFile, permissions);
    } catch (e) {
        log("File " + sandboxFile + " can not be created. This is an error!");
    }
}

//-------------------------------------------------
//assert that there is a directory named 'targetPath'
//if there is one already - does nothing
//if there is not one already - creates it in a sync manner
function checkUserDirExists(targetPath) {
    log("Assert that " + targetPath + " exists");
    try {
        //check if the dir exists
        fs.lstatSync( targetPath);
    } catch(err) {
        //if it doesnt, create it
        if (err.code === 'ENOENT' ) {
            log('No file or directory at ' + targetPath + '. Will create such.');
            try {
                mkdirp.sync(targetPath);
            } catch (err2) {
                if (err2) {
                   log("Cannot create " + targetPath + ". Err " + err.toString());
                } else {
                   log("Dir " + targetPath + " created successfuly");
                }
                
            }
        } else {
            log("checkUserDirExists unknown err " + err);
        }
    }
}

//deletes the filet 'targetPath' if it exists (in a sync manner)
//does nothing if it doesnt exists
function removeFile(targetPath) {
    try {
        log("Trying to delete " + targetPath)
        fs.unlinkSync(targetPath);
    }
    catch(e){
        log("Deleting "  + targetPath + " was not successful " + e.toString());
    }
}

//-------------------------------------------------

//some error codes, because some of us like to stick with C++
//used as a result from the homework() call bellow
//homework returns an object {result:RES_X, txt:"String explaining in a detail why this result happened"}
var RES_OKAY = 0; //everything was okay, we have stored how much tests are passed in the 'txt' param
var RES_INVALID = 1; //some of the input data were not valid
var RES_CANT_WRITE_FILE = 2; //some of the temp files that we had to create failed
var RES_COMPILE_ERROR = 3; //the source did not compile
var RES_TIMEOUT_ERROR = 4; //compilation or execution took too long (> MAX_RUNTIME_MS)

function resToString(res) {
    if (res == RES_OKAY) return "Okay";
    if (res == RES_INVALID) return "RES_INVALID";
    if (res == RES_CANT_WRITE_FILE) return "RES_CANT_WRITE_FILE";
    if (res == RES_COMPILE_ERROR) return "RES_COMPILE_ERROR";
    if (res == RES_TIMEOUT_ERROR) return "RES_TIMEOUT_ERROR";
    else return "UNKOWN RESULT TYPE";
}

//return the folder that holds all the homeworks for the student 'id'
function getUserFolder(id) {
    id = parseFloat(id);
    return HOMEWORK_FOLDER + "/" + id;
}

//returns the folder that holds the 'homewNumber'-th homework for the student 'id'
function getUserHomeworkFolder(id, homeworkNumber) {
    id = parseFloat(id);
    return HOMEWORK_FOLDER + "/" + id + "/" + homeworkNumber.toString();
}

//does the main work
//accepts userid:String, homeworkNumber:Integer and source:String.
//returns {res: RES_X, txt:"string"} RES_X is any of the RES_ numbers defined above
//txt is string explaining why this result happened. If it is a number, it is the time that was spend running the test.
//if res==RES_OKAY, "txt" will have the number of tests that passed successfuly.
//ideally should be split in more functions
function homework(userid, homeworkNumber, source) {
    
    //there is folder that contains the homeworks of all students named HOMEWORK_FOLDER
    //first we assert that such folder exists
    checkUserDirExists(HOMEWORK_FOLDER);

    //to keep in track of the errors in the code (C++ way)
    var result = RES_OKAY;
    
    //sanity check for crappy user ids
    if (!isNumeric(userid)) {
        result = RES_INVALID;
        log("invalid user id " + userid);
    }

    if (result != RES_OKAY)
        return {"result" : result, "txt" : "Invalid user id"};
    
    //this should be number now
    id = parseFloat(userid);
    
    //*************************************************************************
    //* First make sure all the folder/files we will need are available
    
    //make a folder for each student
    //inside that student folder, make a folder for each homework
    var homeworkFolder = getUserHomeworkFolder(id, homeworkNumber);
    var userFolder = getUserFolder(id);
    
    //assert that those folders exsits
    checkUserDirExists(userFolder);
    checkUserDirExists(homeworkFolder);
    
    //create sandbox file that sandboxes to 'userFolder'
    createSandboxFile(userFolder);
    
    //make a .cpp file that will store the homework
    var homeworkFile = homeworkFolder + "/" + HOMEWORK_EXECUTABLE_NAME + ".cpp";
    
    //clear any previous files just in case
    removeFile(homeworkFile);
    
    //write the source in the file
    try {
        fs.writeFileSync(homeworkFile, source);
    } catch(e) {
        return {"result" : RES_CANT_WRITE_FILE, "txt" :  "Can not create file"};
    }
    
    //get path to the executable
    var outFile = homeworkFolder + "/" + HOMEWORK_EXECUTABLE_NAME + ".bin";
    //clear any previous files just in case
    removeFile(outFile);
    
    //*************************************************************************
    //Call clang to compile the source to binary
    
    //limit the executable file resources (time & memory)
    var processOptions  = { encoding: 'utf8',
            timeout: 50,
            maxBuffer: 200*1024,
            killSignal: 'SIGTERM',
            cwd: null,
            env: null
    };
    
    //call clang to compile the source
    try {
        var compileResult = execSync(COMPILER_START_COMMAND + ' ' + homeworkFile + " -o " + outFile, MAX_RUNTIME_MS);
    } catch (e) {
        result = RES_COMPILE_ERROR;
        return {"result" : result, "txt" : e.toString()};
    }
    
    //check if the compilation is okay
    if (compileResult.stderr) {
        result = RES_COMPILE_ERROR;
        return {"result" : result, "txt" : compileResult.stderr.toString()}
    }
    
    //*************************************************************************
    //Call the compiled binary once for each of the tests
    
    var numTestsPassed = 0;
    log("homeworkFolder " + homeworkFolder);
    var timeTaken = 0;

    //loop through the test cases and compare the results
    for (var i = 0; i < NUM_RUNS; ++i) {
        var timeMs = 0;
        try {
             //execute the compiled binary
            var execCommand = "";
            if (USE_SANDBOXING) {
                 execCommand = "sudo sandbox-exec -f " + userFolder + "/sandbox " + outFile;
            } else {
                 execCommand = "./" + outFile + " " + input;
            }
            
             log("Executing '" + execCommand + "'");
             timeMs = process.hrtime();

             var chModResult = execSync(execCommand, MAX_RUNTIME_MS);
        } catch (e) {
            log("chroot err " + e.toString());
            if (USE_SANDBOXING) {
                execSync("sudo pkill -9 " + HOMEWORK_EXECUTABLE_NAME + ".bin")
            }
            
            return {"result":RES_TIMEOUT_ERROR, "txt":"Time out error"};
        }
        
        //check how much time the executable was running
        var diff = process.hrtime(timeMs);
        timeTaken += (diff[0] * 1e9 + diff[1]) / 1000000;
        
        //get how much time it is allowed to take
        //var timeMax = parseFloat(tests[i].time.trim());
        //fetch the student result & the needed result
        
        var homeworkResult = chModResult.stdout.toString().trim();
    
        if (homeworkResult.indexOf("BAD") >  -1) {
            return {"result":RES_INVALID, "txt":"bad result"};
        }
        //var output = tests[i].output.trim();
        log(outFile + " result stdout=\'" + chModResult.stdout + "' stderr='" + chModResult.stderr + ", time:" + timeTaken + "'");
    }
    //*************************************************************************
    //Hopefully nobody hacked us and we can return the result
    
    return {"result" : result, "txt":timeTaken/NUM_RUNS};
}


//walk on all the files with student solutions and get the paths to them
var walker  = walk.walk(SOLUTIONS_FOLDER, { followLinks: false });

//stores paths to the .cpp files with the student solutions
var files   = [];

walker.on('file', function(root, stat, next) {
          // Add this file to the list of files
          if (stat.name.indexOf(".DS") < 0)
              files.push(root + '/' + stat.name);
          next();
});

//search for the `teacher` solution against which student ones will be tested
var pathToSolution = "./solution" + HOMEWORK_NUMBER.toString() + ".cpp";
log("Searching for solution in " + pathToSolution);
var baseSolution = "";
try {
    baseSolution = fs.readFileSync(pathToSolution).toString();
} catch (e) {
    log("Error loading solution file " + pathToSolution + ". Script will abort");
    process.exit(1);
}

//sorts array by key
function sortByKey(array, key) {
    return array.sort(function(a, b) {
                      var x = a[key]; var y = b[key];
                      return ((x < y) ? -1 : ((x > y) ? 1 : 0));
                      });
}

//go through each .cpp file, compile, run, check for invalid result (f.e. if "BAD" will be printed or if it will crash / hang / etc) and report times
walker.on('end', function() {
  var runString = "run";
  if (NUM_RUNS > 1) runString = "runs";
  var messages = "\n\n\n ******************* RESULTS (" + NUM_RUNS + " " + runString + ") *******************\n";

    var results = [];
    for (var i = 0; i < files.length; ++i) {
        try {

            var path = files[i];
            var fnum = path.substring(SOLUTIONS_FOLDER.length + 1, path.length);
            var id = parseFloat(fnum);
            var src = fs.readFileSync(path).toString();

            src = src.split(" main(").join(" _3bb005aa__fmi_hpc_2015_main2_invalidated__(");

            src += baseSolution;

            log("\n\n **** Testing solution of student " + id + " ****")

            var res = homework(id, HOMEWORK_NUMBER, src);

            results.push( {"result":resToString(res.result), "avgTime":res.txt, "facultyNumber":id} );

          } catch(e) {
            messages += ("Error executing: " + id + " " + e) + '\n';
        }
    }
          
    sortByKey(results, "avgTime");

    for (var i = 0; i < results.length; ++i) {
        var result = results[i];
        messages += "Faculty number: " + result.facultyNumber + ", result:" + result.result + ", avg time:" + result.avgTime + "\n";
    }
          
    log(messages);

});
