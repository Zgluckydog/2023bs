$(function() {
    // 监听表单的提交事件
    $("#high-form").submit(function(event) {
        event.preventDefault();  // 阻止表单提交
        // 从表单中获取数据
        let shiwen2 = $("#shiwen2").val();
        let jiwen2 = $("#jiwen2").val();
        let zhuansu2 = $("#zhuansu2").val();
        let niuju2 = $("#niuju2").val();
        let gonglv2 = $("#gonglv2").val();
        // 发送POST请求到服务器
        $.ajax({
            url: "/predict_from_high",
            type: "POST",
            data: {
                "shiwen2": shiwen2,
                "jiwen2": jiwen2,
                "zhuansu2": zhuansu2,
                "niuju2": niuju2,
                "gonglv2": gonglv2

            },
            success: function(response) {
                // 在页面中显示预测结果
                let prediction = response.prediction[0];
                let resultDiv = $("#result_high");
                resultDiv.html("Prediction: " + prediction);
            },
            error: function(xhr, status, error) {
                // 显示错误信息
                let resultDiv = $("#result_high");
                resultDiv.html("Error: " + error);
            }
        });
    });
    // 监听文件上传表单的提交事件
    $("#highfile-form").submit(function(event) {
        event.preventDefault();  // 阻止表单提交
        // 获取文件数据
        let file = $("input[name='file_high']").prop("files")[0];
        let formData = new FormData();
        formData.append("file", file);
        // 发送POST请求到服务器
        $.ajax({
            url: "/predict_from_highfile",
            type: "POST",
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                // 在页面中显示预测结果
                let resultDiv = $("#file-result_high");
                resultDiv.html("Predictions: " + response.predictions.join(", "));
            },
            error: function(xhr, status, error) {
                // 显示错误信息
                let resultDiv = $("#file-result_high");
                resultDiv.html("Error: " + error);
            }
        });
    });
});