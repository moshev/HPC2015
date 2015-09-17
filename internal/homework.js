// Simple node js script that can be used to compile, run and evaluate the result of C++ files
// Check sampleUsage() at the end of the file for example usage

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

var USE_SANDBOXING = (os.platform()=="darwin") && parseFloat(os.release()) >= 15;

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
var HOMEWORK_FOLDER = "homeworks_main";

//maximum amount of runtime for each homework
//the same amount is used as max amount of time for clang during homework compilation
var MAX_RUNTIME_MS = 10000;

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

//creates file named "sandbox" in the current directory, which has the configuration for running ./sandbox-exec.
//This allows process started with "./sandbox-exec -f sandbox /path/to/process/process.bin" to mess up
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
//accepts userid:String, homeworkNumber:Integer, Tests:[{input:"string", output:"string", time:Number},...] and source:String.
//returns {res: RES_X, txt:"string"} RES_X is any of the RES_ numbers defined above
//txt is string explaining why this result happened
//if res==RES_OKAY, "txt" will have the number of tests that passed successfuly.
//ideally should be split in more functions
function homework(userid, homeworkNumber, tests, source) {
    
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
            timeout: 10,
            maxBuffer: 200*1024,
            killSignal: 'SIGTERM',
            cwd: null,
            env: null
    };
    
    //call clang to compile the source
    try {
        var compileResult = execSync('clang++ -Xclang -std=c++11 -stdlib=libc++ -lpthread  ' + homeworkFile + " -o " + outFile, MAX_RUNTIME_MS);
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
    //loop through the test cases and compare the results
    for (var i = 0; i < tests.length; ++i) {
        
        //fetch the input params
        var input = tests[i].input.trim();
        
        var timeMs = 0;
        try {
             //execute the compiled binary
            var execCommand = "";
            if (USE_SANDBOXING) {
                 execCommand = "sudo sandbox-exec -f " + userFolder + "/sandbox " + outFile + " " + input;
            } else {
                 execCommand = "./" + outFile + " " + input;
            }
            
             log("Executing '" + execCommand + "'");
             timeMs = process.hrtime();

             var chModResult = execSync(execCommand, MAX_RUNTIME_MS);
        } catch (e) {
            log("chroot err " + e.toString());
            execSync("sudo pkill -9 " + HOMEWORK_EXECUTABLE_NAME + ".bin")
            return {"result":RES_TIMEOUT_ERROR, "txt":"Time out error"};
        }
        
        //check how much time the executable was running
        var diff = process.hrtime(timeMs);
        timeTaken = (diff[0] * 1e9 + diff[1]) / 1000000;
        
        //get how much time it is allowed to take
        var timeMax = parseFloat(tests[i].time.trim());
        //fetch the student result & the needed result
        
        var homeworkResult = chModResult.stdout.toString().trim();
        var output = tests[i].output.trim();
        log(outFile + " result stdout=\'" + chModResult.stdout + "' stderr='" + chModResult.stderr + ", time:" + timeTaken + "'");
        
        //check if the result is correct
        if (output == homeworkResult) {
            
            //and within time limit
            if (timeTaken > timeMax) {
                log("Test " + i + " for id " + id + " FAILED (TIME OUT)");
            } else {
                log("Test " + i + " for id " + id + " OKAY (" + output + " == " + homeworkResult + ")");
                numTestsPassed++;
            }
        } else {
            log("Test " + i + " for id " + id + " FAILED (" + output + " != " + homeworkResult + ")");
        }
        
    }
    
    //*************************************************************************
    //Hopefully nobody hacked us and we can return the result
    
    return {"result" : result, "txt" : numTestsPassed.toString()};
}

function sampleUsage() {
    var testSuite = [{input:"0 2 0", output:"1 1 1", time:"300"},
                     {input:"1 0 0", output:"1 2 1 ", time:"1000"},
                     {input:"0 0 3", output:"3 1 1 ", time:"2000"}];
    
    for (var i = 0; i < 1; ++i) {
        for (var j = 0; j < 1; ++j) {
            var studentId = 44286+i;
            var homeworkNumber = j;
            var source = '#include <vector>\n#include <unistd.h>\n#include <stdio.h>\n#include <iostream>\n int main(int argc, const char* argv[]){std::vector<int> v;printf("1 1 1");}';
            
            //******************************************
            var res = homework(studentId.toString(), //*
                               homeworkNumber,       //*
                               testSuite,            //*
                               source);              //*
            //******************************************

            var resultMessage = ""
            if (res.result == RES_OKAY) {
                resultMessage = "****** Tests for homework#" + homeworkNumber + ", studendId " + studentId + ". Passed " + res.txt + " out of " + testSuite.length + " tests.***";
            } else {
                resultMessage = "****** Tests for homework#" + homeworkNumber + ", studendId " + studentId + ". Failed, reason: [" + res.txt + "]";
            }
            
            log(resultMessage);
            
            var resultFile = getUserHomeworkFolder(studentId, homeworkNumber) + "/result.txt";
            removeFile(resultFile);
            try {
                fs.writeFileSync(resultFile, resultMessage);
            } catch(e) {
                log("Can't create result file " + resultFile);
            }
        }
    }
   
}

sampleUsage();