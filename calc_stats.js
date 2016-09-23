
var fs = require("fs")

fs.readFile(process.argv[2], function (err, data) {

    var stats = {}

    var output = data.toString();
    var re = /\n==== ([\s\S]+?) ====[^=]+Elapsed: ([\.\d]+)ms/g
    outputs = output.split("SIZE: ");

    for (var i = 1; i < outputs.length; ++i) {
        var out = outputs[i];
        var size = parseFloat(out.match(/\d+/)[0])
        var match = re.exec(out)
        while (match != null) {

            if (!(size in stats)) {
                console.log('initing', size)
                stats[size] = new Object()
            }
            if (!(match[1] in stats[size])) {
                console.log('initing', size, match[1])
                stats[size][match[1]] = [0, 0]
            }
            stats[size][match[1]][0] += 1
            stats[size][match[1]][1] += parseFloat(match[2])

            match = re.exec(out)
        }
    }
    for (var i in stats) {
        for (var j in stats[i]) {
            stats[i][j] = stats[i][j][1] / stats[i][j][0]
        }
    }
    console.log(stats)
    
})