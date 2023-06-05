$(function() {
    // 监听表单的提交事件
    $("#med-form").submit(function(event) {
        event.preventDefault();  // 阻止表单提交
        // 从表单中获取数据
        let shiwen = $("#shiwen").val();
        let jiwen = $("#jiwen").val();
        let zhuansu1 = $("#zhuansu1").val();
        let niuju1 = $("#niuju1").val();
        let gonglv1 = $("#gonglv1").val();
        let shichang1 = $("#shichang1").val();
        // 发送POST请求到服务器
        $.ajax({
            url: "/predict_from_med",
            type: "POST",
            data: {
                "shiwen": shiwen,
                "jiwen": jiwen,
                "zhuansu1": zhuansu1,
                "niuju1": niuju1,
                "gonglv1": gonglv1,
                "shichang1": shichang1

            },
            success: function(response) {
                // 在页面中显示预测结果
                let prediction = response.prediction[0];
                let resultDiv = $("#result_med");
                resultDiv.html("Prediction: " + prediction);
            },
            error: function(xhr, status, error) {
                // 显示错误信息
                let resultDiv = $("#result_med");
                resultDiv.html("Error: " + error);
            }
        });
    });
    // 监听文件上传表单的提交事件
    $("#medfile-form").submit(function(event) {
        event.preventDefault();  // 阻止表单提交
        // 获取文件数据
        let file = $("input[name='file_med']").prop("files")[0];
        let formData = new FormData();
        formData.append("file", file);
        // 发送POST请求到服务器
        $.ajax({
            url: "/predict_from_medfile",
            type: "POST",
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                // 在页面中显示预测结果
                let resultDiv = $("#file-result_med");
                resultDiv.html("Predictions: " + response.predictions.join(", "));
            },
            error: function(xhr, status, error) {
                // 显示错误信息
                let resultDiv = $("#file-result_med");
                resultDiv.html("Error: " + error);
            }
        });
    });
});