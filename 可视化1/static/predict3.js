$(function() {
    // 监听表单的提交事件
    $("#low-form").submit(function(event) {
        event.preventDefault();  // 阻止表单提交
        // 从表单中获取数据
        let niuju = $("#niuju").val();
        let gonglv = $("#gonglv").val();
        let shichang = $("#shichang").val();
        let zhuansu = $("#zhuansu").val();
        // 发送POST请求到服务器
        $.ajax({
            url: "/predict_from_low",
            type: "POST",
            data: {
                "niuju": niuju,
                "gonglv": gonglv,
                "shichang": shichang,
                "zhuansu": zhuansu
            },
            success: function(response) {
                // 在页面中显示预测结果
                let prediction = response.prediction[0];
                let resultDiv = $("#result_low");
                resultDiv.html("Prediction: " + prediction);
            },
            error: function(xhr, status, error) {
                // 显示错误信息
                let resultDiv = $("#result_low");
                resultDiv.html("Error: " + error);
            }
        });
    });
    // 监听文件上传表单的提交事件
    $("#lowfile-form").submit(function(event) {
        event.preventDefault();  // 阻止表单提交
        // 获取文件数据
        let file = $("input[name='file_low']").prop("files")[0];
        let formData = new FormData();
        formData.append("file", file);
        // 发送POST请求到服务器
        $.ajax({
            url: "/predict_from_lowfile",
            type: "POST",
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                // 在页面中显示预测结果
                let resultDiv = $("#file-result_low");
                resultDiv.html("Predictions: " + response.predictions.join(", "));
            },
            error: function(xhr, status, error) {
                // 显示错误信息
                let resultDiv = $("#file-result_low");
                resultDiv.html("Error: " + error);
            }
        });
    });
});