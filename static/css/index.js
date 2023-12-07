$(document).ready(function () {
    $("#prediction").hide();  // Hide the message initially

    $("#name-form").submit(function (event) {
        event.preventDefault();
        $.ajax({
            type: "POST",
            url: "/check_price",
            data: $("#name-form").serialize(),
            success: function (response) {  
                $("#pred-res").text(response.result);
                console.log(response.arr);
                $("#prediction").show();
            }
        });
    });

    $("#brand").change(()=>{
        var brand = $('#brand').val();
        // var option = $("#model").html();
        var option = "<option>Select Model</option>";
        
        $.ajax({
            type: 'POST',
            url: '/brand_name',
            contentType: 'application/json',
            data: JSON.stringify({'brand': brand}),
            success: function(data) {
                // console.log(data);
                data.sort();
                data.forEach((b)=>{
                    option = option + '<option value='+ b + '>' + b + '</option>';
                })
                $("#model").html(option);    
            },
            error: function(error) {
                // $('#cityDetails').html(`<p>${error.responseJSON.error}</p>`);
                console.log(error);
                
            }
        });      
    });
});

